"""Restore one subdirectory out of a read-only ``tar.gz`` archive.

Guarantees
----------
* The archive file is opened exactly once per request.  Both passes read from
  that same open file description, so an atomic rotation of the path between
  the passes cannot mix bytes from the old and the new backup: a request that
  opened the old file finishes from the old file, while a request that starts
  after the rotation uses the new one.
* Decompression is done in chunks by :class:`_ValidatedGzipReader`, which uses
  :class:`gzip.GzipFile` underneath and therefore verifies the CRC32 *and*
  ISIZE trailer of **every** gzip member.  Concatenated (multi-member) gzip
  streams are fully supported.
* The entire gzip stream is validated before anything is published, even when
  the selected directory appeared early in the archive.  A truncated tail or a
  bad member checksum fails the restore and removes this attempt's staging
  area.
* Only the selected subtree is materialised.  Bytes of members that are not
  needed are read in chunks and discarded, never written.
* The total number of *extracted* payload bytes is at most twice the payload of
  one complete decompression.  There is exactly one full scan (1x), including
  the gzip tail; a second scan happens only when a retained hard link points at
  a regular file outside the selection, and it stops as soon as those targets
  have been read (<= 1x).
* A hard link whose target lies outside the selection, appears later in the
  archive, or is itself a hard link (chained) is resolved against the archive
  root.  All retained entries that resolve to the same regular-file member are
  hard-linked together, so they share one inode.
* Missing / cyclic / non-regular link chains that a retained entry depends on
  raise :class:`LinkError`.
* Restoration is staged in a hidden directory; the final directory is only
  published with one atomic ``rename`` after every byte and every hard link is
  in place.  Any failure removes everything produced by that attempt.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tarfile
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB

DIR = "dir"
FILE = "file"
HARDLINK = "hardlink"
SYMLINK = "symlink"
OTHER = "other"


class RestoreError(Exception):
    """Base class for all restore failures."""


class InvalidSelection(RestoreError):
    """The requested subdirectory path is not a safe relative path."""


class PathNotFound(RestoreError):
    """The requested subdirectory is not present in the archive."""


class NotADirectory(RestoreError):
    """The requested path exists but is not a directory."""


class LinkError(RestoreError):
    """A retained hard link is missing, cyclic or points at a non-file."""


class MalformedArchive(RestoreError):
    """The archive contains conflicting or otherwise unusable members."""


class InvalidArchive(RestoreError):
    """The archive cannot be read as a gzip-compressed tar stream."""


@dataclass
class Member:
    name: str
    kind: str
    size: int = 0
    mode: int = 0o644
    mtime: float = 0.0
    linkname: Optional[str] = None


@dataclass
class RestoreResult:
    requested_path: str
    restored_path: str = ""
    directories: int = 1  # the restored root itself
    files: int = 0
    hardlinks: int = 0
    bytes_written: int = 0
    bytes_extracted: int = 0
    # Every byte produced by the gzip decoder, across all passes, including
    # members that were skipped and the tail that only existed to be checked.
    bytes_decompressed: int = 0
    archive_payload_bytes: int = 0
    skipped_unsafe: int = 0
    skipped_unsupported: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_member_name(raw: str) -> str:
    """Normalise an archive member path, rejecting anything that escapes it."""

    value = raw.replace("\\", "/")
    if value.startswith("/"):
        raise MalformedArchive(f"absolute member path rejected: {raw!r}")
    parts = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise MalformedArchive(f"member path escapes archive root: {raw!r}")
        parts.append(part)
    return "/".join(parts)


def normalize_selection(raw: str) -> str:
    """Normalise the user-supplied subdirectory path."""

    if not isinstance(raw, str) or not raw.strip():
        raise InvalidSelection("path must be a non-empty relative directory path")
    value = raw.strip().replace("\\", "/")
    if value.startswith("/"):
        raise InvalidSelection("path must be relative to the archive root")
    parts = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise InvalidSelection("path must not contain '..'")
        parts.append(part)
    if not parts:
        raise InvalidSelection("path must select a subdirectory, not the archive root")
    return "/".join(parts)


def _within(path: str, root: str) -> bool:
    return bool(root) and (path == root or path.startswith(root + "/"))


def _rel(path: str, root: str) -> str:
    return path[len(root) + 1 :]


def _member_kind(info: tarfile.TarInfo) -> str:
    if info.isdir():
        return DIR
    if info.isreg():
        return FILE
    if info.islnk():
        return HARDLINK
    if info.issym():
        return SYMLINK
    return OTHER


def _open_stream(
    fileobj: BinaryIO, stats: Optional[RestoreResult] = None
) -> tuple[tarfile.TarFile, "_ValidatedGzipReader"]:
    """Open a tar stream over ``fileobj`` and validate gzip down to the last bit.

    ``fileobj`` is already open on the backup, so the same file description is
    used for every pass.  ``tarfile`` is fed *decompressed* bytes (``r|``); all
    gzip framing, CRC32 and ISIZE checks are performed by the reader below.
    """

    try:
        fileobj.seek(0)
    except (OSError, ValueError) as exc:
        raise InvalidArchive(f"cannot rewind archive: {exc}") from exc
    reader = _ValidatedGzipReader(fileobj, stats)
    try:
        stream = tarfile.open(fileobj=reader, mode="r|")
    except (tarfile.TarError, EOFError, zlib.error, OSError, InvalidArchive) as exc:
        reader.close()
        if isinstance(exc, InvalidArchive):
            raise
        raise InvalidArchive(f"cannot open archive: {exc}") from exc
    return stream, reader


class _ValidatedGzipReader:
    """Read a (possibly multi-member) gzip stream and verify every member.

    ``gzip.GzipFile`` validates the CRC32 and ISIZE trailer of each member as
    its end is reached and transparently continues with the next member.
    Reading until it returns ``b""`` is therefore what proves the *whole*
    stream is intact; :meth:`finish` does exactly that for the caller.
    """

    def __init__(
        self, fileobj: BinaryIO, stats: Optional[RestoreResult] = None
    ) -> None:
        self.stats = stats
        self.decompressed = 0
        try:
            self._gz = gzip.GzipFile(fileobj=fileobj, mode="rb")
        except (OSError, EOFError, zlib.error) as exc:
            raise InvalidArchive(f"not a valid gzip stream: {exc}") from exc

    def read(self, size: int = -1) -> bytes:
        try:
            chunk = self._gz.read(size)
        except gzip.BadGzipFile as exc:
            # CRC32 or ISIZE mismatch in some member's trailer.
            raise InvalidArchive(f"gzip member check failed: {exc}") from exc
        except (EOFError, zlib.error) as exc:
            raise InvalidArchive(f"corrupt or truncated gzip stream: {exc}") from exc
        self.decompressed += len(chunk)
        if self.stats is not None:
            self.stats.bytes_decompressed += len(chunk)
        return chunk

    def finish(self) -> None:
        """Consume the remainder of the stream, validating its tail."""

        while True:
            if not self.read(CHUNK_SIZE):
                return

    def close(self) -> None:
        try:
            self._gz.close()
        except OSError:
            pass


def _drain(reader: Optional[BinaryIO], stats: RestoreResult) -> None:
    if reader is None:
        return
    while True:
        chunk = reader.read(CHUNK_SIZE)
        if not chunk:
            return
        stats.bytes_extracted += len(chunk)


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _file_mode(mode: int) -> int:
    # Keep the archived permissions but always let the owner read/write, so the
    # restored data is usable and can be cleaned up on failure.
    return (mode & 0o777) | 0o600


def _dir_mode(mode: int) -> int:
    # Owner needs rwx to traverse the tree; without the execute bit a directory
    # cannot be descended into (and even cleanup would fail).
    return (mode & 0o777) | 0o700


def _write_payload(
    reader: BinaryIO,
    dest: str,
    member: Member,
    stats: RestoreResult,
    fsync: bool,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(dest, flags, 0o600)
    written = 0
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            stats.bytes_extracted += len(chunk)
            out.write(chunk)
            written += len(chunk)
        out.flush()
        os.fchmod(out.fileno(), _file_mode(member.mode))
        if fsync:
            os.fsync(out.fileno())
    if member.mtime > 0:
        try:
            os.utime(dest, (member.mtime, member.mtime), follow_symlinks=False)
        except OSError:
            pass
    stats.bytes_written += written


class _OutputTree:
    """Tracks the staging tree and defends against path/type conflicts."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.dir_paths: set[str] = {""}
        self.file_paths: set[str] = set()
        self.dir_meta: dict[str, tuple[int, float]] = {}

    def dest(self, rel: str) -> str:
        if not rel:
            return self.root
        if rel.startswith("/") or ".." in rel.split("/"):
            raise MalformedArchive(f"refusing path outside restore root: {rel!r}")
        return os.path.join(self.root, *rel.split("/"))

    def ensure_dir(self, rel: str, mode: Optional[int] = None, mtime: float = 0.0) -> None:
        if rel in self.file_paths:
            raise MalformedArchive(f"path {rel!r} is both a file and a directory")
        if rel and rel not in self.dir_paths:
            # Implicit parents (archives do not always list every directory).
            self.ensure_parents(rel)
        if rel not in self.dir_paths:
            try:
                os.mkdir(self.dest(rel), 0o700)
            except FileExistsError:
                path = self.dest(rel)
                if os.path.islink(path) or not os.path.isdir(path):
                    raise MalformedArchive(f"path {rel!r} conflicts with an existing entry")
            self.dir_paths.add(rel)
        if mode is not None:
            self.dir_meta[rel] = (mode, mtime)

    def ensure_parents(self, rel: str) -> None:
        if not rel:
            return
        parts = rel.split("/")[:-1]
        current = ""
        for part in parts:
            current = part if not current else f"{current}/{part}"
            self.ensure_dir(current)

    def register_file(self, rel: str) -> None:
        if not rel:
            raise MalformedArchive("refusing to write the restore root as a file")
        if rel in self.dir_paths or rel in self.file_paths:
            raise MalformedArchive(f"duplicate or conflicting member {rel!r}")
        self.file_paths.add(rel)

    def finalize_dirs(self, fsync: bool) -> None:
        for rel in sorted(self.dir_paths, key=lambda p: p.count("/"), reverse=True):
            path = self.dest(rel)
            mode, mtime = self.dir_meta.get(rel, (0o755, 0.0))
            try:
                os.chmod(path, _dir_mode(mode))
                if mtime > 0:
                    os.utime(path, (mtime, mtime))
                if fsync:
                    _fsync_dir(path)
            except OSError as exc:  # pragma: no cover - unusual filesystem
                logger.warning("could not finalise directory %s: %s", path, exc)


def _resolve_hardlink(
    name: str,
    members: dict[str, Member],
    cache: dict[str, str],
) -> str:
    """Return the archive path of the regular file that owns ``name``'s data."""

    chain: list[str] = []
    seen: set[str] = set()
    current = name
    while True:
        if current in cache:
            source = cache[current]
            for step in chain:
                cache[step] = source
            return source
        if current in seen:
            start = chain.index(current)
            cycle = " -> ".join(chain[start:] + [current])
            raise LinkError(f"cyclic hard link chain required by {name!r}: {cycle}")
        seen.add(current)
        chain.append(current)

        member = members.get(current)
        if member is None:
            raise LinkError(
                f"hard link target {current!r} is missing from the archive"
                f" (required by {' -> '.join(chain)})"
            )
        if member.kind == FILE:
            for step in chain:
                cache[step] = current
            return current
        if member.kind != HARDLINK:
            raise LinkError(
                f"hard link target {current!r} is a {member.kind}, not a regular file"
                f" (required by {' -> '.join(chain)})"
            )
        if not member.linkname:
            raise LinkError(f"hard link {current!r} has an empty target")
        current = member.linkname


@dataclass
class _Plan:
    groups: dict[str, list[str]] = field(default_factory=dict)  # source -> selected names
    canonical: dict[str, str] = field(default_factory=dict)  # source -> selected name
    external: dict[str, str] = field(default_factory=dict)  # source -> canonical dest rel

    def destination_rel(self, source: str, selection: str) -> str:
        """Path (relative to the restored root) holding ``source``'s data."""

        if source in self.external:
            return self.external[source]
        return _rel(source, selection)


def _populate(
    archive_file: BinaryIO,
    selection: str,
    output_root: str,
    stats: RestoreResult,
    fsync: bool,
) -> _Plan:
    tree = _OutputTree(output_root)
    members: dict[str, Member] = {}
    selected_files: set[str] = set()
    selected_links: list[str] = []
    plan = _Plan()
    selection_member: Optional[Member] = None
    has_descendants = False

    stream, gz_reader = _open_stream(archive_file, stats)
    try:
        with stream:
            for info in stream:
                try:
                    name = normalize_member_name(info.name)
                except MalformedArchive as exc:
                    logger.warning("skipping unsafe member: %s", exc)
                    stats.skipped_unsafe += 1
                    if info.isreg():
                        stats.archive_payload_bytes += info.size
                        _drain(stream.extractfile(info), stats)
                    continue
                if not name:
                    if info.isreg():
                        stats.archive_payload_bytes += info.size
                        _drain(stream.extractfile(info), stats)
                    continue
                kind = _member_kind(info)
                if name in members:
                    existing = members[name]
                    # Duplicate directory entries occur in real archives (e.g. an
                    # explicit "dir/" plus an implicit one); treat them as one.
                    if kind == DIR and existing.kind == DIR:
                        continue
                    raise MalformedArchive(f"duplicate archive member {name!r}")

                linkname = None
                invalid_link = None
                if kind == HARDLINK:
                    try:
                        linkname = normalize_member_name(info.linkname or "")
                        if not linkname:
                            invalid_link = "empty target"
                    except MalformedArchive as exc:
                        invalid_link = str(exc)
                    if invalid_link and _within(name, selection):
                        # A link we must reproduce cannot be trusted; fail loudly.
                        raise LinkError(f"hard link {name!r}: {invalid_link}")
                member = Member(
                    name=name,
                    kind=kind,
                    size=info.size if kind == FILE else 0,
                    mode=info.mode,
                    mtime=info.mtime,
                    linkname=linkname,
                )
                members[name] = member

                if kind == FILE:
                    stats.archive_payload_bytes += info.size

                inside = _within(name, selection)
                if name == selection:
                    selection_member = member
                if name.startswith(selection + "/"):
                    has_descendants = True

                if kind == FILE:
                    if inside and name != selection:
                        rel = _rel(name, selection)
                        tree.ensure_parents(rel)
                        tree.register_file(rel)
                        reader = stream.extractfile(info)
                        if reader is None:
                            raise MalformedArchive(f"cannot read payload of {name!r}")
                        _write_payload(reader, tree.dest(rel), member, stats, fsync)
                        selected_files.add(name)
                        stats.files += 1
                    else:
                        _drain(stream.extractfile(info), stats)
                elif kind == DIR:
                    if inside and name != selection:
                        rel = _rel(name, selection)
                        tree.ensure_dir(rel, mode=member.mode, mtime=member.mtime)
                elif kind == HARDLINK:
                    if inside and name != selection:
                        rel = _rel(name, selection)
                        tree.ensure_parents(rel)
                        tree.register_file(rel)
                        selected_links.append(name)
                        stats.hardlinks += 1
                elif inside and name != selection:
                    # Symlinks, devices, fifos: out of scope, never materialised, so
                    # they cannot be used to escape the staging tree.
                    stats.skipped_unsupported += 1

        # The selected directory may have ended early in the archive; the rest
        # of the gzip stream (including its trailer) is still read and checked.
        gz_reader.finish()
        full_decompressed = gz_reader.decompressed
    finally:
        gz_reader.close()

    if selection_member is not None and selection_member.kind != DIR:
        raise NotADirectory(
            f"{selection!r} is a {selection_member.kind}, not a directory"
        )
    if selection_member is None and not has_descendants:
        raise PathNotFound(f"directory {selection!r} not found in the archive")

    link_cache: dict[str, str] = {}
    for name in selected_links:
        source = _resolve_hardlink(name, members, link_cache)
        plan.groups.setdefault(source, []).append(name)
    for name in selected_files:
        plan.groups.setdefault(name, []).append(name)

    for source, entries in plan.groups.items():
        if source in selected_files:
            # The owning regular file is itself part of the selection: it is
            # already on disk, so every link can point at it.
            plan.canonical[source] = source
        else:
            # Only links to an outside file were selected. Materialise the
            # payload under one of the selected names; the rest link to it.
            canonical = sorted(entries)[0]
            plan.canonical[source] = canonical
            plan.external[source] = _rel(canonical, selection)

    if plan.external:
        _fetch_external(archive_file, members, plan, tree, stats, fsync)

    # Create the hard links only after every payload exists, so forward and
    # chained references are all resolvable.
    for source, entries in plan.groups.items():
        canonical_path = tree.dest(plan.destination_rel(source, selection))
        for entry in entries:
            if entry == plan.canonical[source]:
                continue
            rel = _rel(entry, selection)
            try:
                os.link(canonical_path, tree.dest(rel))
            except OSError as exc:
                raise RestoreError(f"cannot create hard link {rel!r}: {exc}") from exc

    if selection_member is not None:
        tree.dir_meta[""] = (selection_member.mode, selection_member.mtime)
    tree.finalize_dirs(fsync)
    stats.directories = len(tree.dir_paths)

    # Budget: the first pass proved the whole stream (full_decompressed); the
    # optional second pass is a prefix of it.  The cumulative decompressed
    # volume therefore never exceeds a complete decompression twice.
    if stats.bytes_decompressed > 2 * full_decompressed:
        raise RestoreError(
            "internal error: decompression budget exceeded "
            f"({stats.bytes_decompressed} > 2 x {full_decompressed})"
        )
    if stats.bytes_extracted > 2 * stats.archive_payload_bytes:
        raise RestoreError(
            "internal error: extracted byte budget exceeded "
            f"({stats.bytes_extracted} > 2 x {stats.archive_payload_bytes})"
        )
    return plan


def _fetch_external(
    archive_file: BinaryIO,
    members: dict[str, Member],
    plan: _Plan,
    tree: _OutputTree,
    stats: RestoreResult,
    fsync: bool,
) -> None:
    """Second pass: copy link targets that live outside the selection.

    This pass may stop as soon as every needed payload has been read: the
    first pass already consumed and validated the whole gzip stream, so the
    tail does not need to be re-checked here.
    """

    remaining = set(plan.external)
    if not remaining:
        return
    stream, gz_reader = _open_stream(archive_file, stats)
    try:
        with stream:
            for info in stream:
                if not info.isreg():
                    continue
                try:
                    name = normalize_member_name(info.name)
                except MalformedArchive:
                    _drain(stream.extractfile(info), stats)
                    continue
                if name in remaining:
                    canonical_rel = plan.external[name]
                    tree.ensure_parents(canonical_rel)
                    reader = stream.extractfile(info)
                    if reader is None:
                        raise MalformedArchive(f"cannot read payload of {name!r}")
                    _write_payload(
                        reader, tree.dest(canonical_rel), members[name], stats, fsync
                    )
                    remaining.remove(name)
                    if not remaining:
                        break
                else:
                    _drain(stream.extractfile(info), stats)
    finally:
        gz_reader.close()
    if remaining:
        raise LinkError(
            "hard link target(s) could not be located during the data pass: "
            + ", ".join(sorted(remaining))
        )


def _cleanup(paths: list[str]) -> None:
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def _publish(
    token_dir: str,
    restore_root: str,
    restore_id: str,
    basename: str,
    fsync: bool,
) -> str:
    # ``token_dir`` contains exactly the restored directory (basename). Renaming
    # the token directory publishes the whole result in one atomic step, so
    # nothing partial is ever visible under ``restore_root``.
    final_parent = os.path.join(restore_root, restore_id)
    os.rename(token_dir, final_parent)
    final_path = os.path.join(final_parent, basename)
    if fsync:
        _fsync_dir(final_parent)
        _fsync_dir(restore_root)
    return final_path


def restore_directory(
    archive_path: str,
    selection: str,
    restore_root: str,
    *,
    fsync: Optional[bool] = None,
) -> RestoreResult:
    """Restore ``selection`` from ``archive_path`` and publish it atomically."""

    selection = normalize_selection(selection)
    if fsync is None:
        fsync = os.environ.get("RESTORE_FSYNC", "1").lower() not in ("0", "false", "no")
    if not os.path.isfile(archive_path):
        raise PathNotFound(f"archive {archive_path!r} does not exist")

    restore_root = os.path.abspath(restore_root)
    os.makedirs(restore_root, exist_ok=True)
    staging_root = os.path.join(restore_root, ".staging")
    os.makedirs(staging_root, mode=0o700, exist_ok=True)

    token = uuid.uuid4().hex
    restore_id = uuid.uuid4().hex
    basename = selection.rsplit("/", 1)[-1]
    token_dir = os.path.join(staging_root, token)
    output_root = os.path.join(token_dir, basename)
    stats = RestoreResult(requested_path=selection)

    try:
        os.makedirs(token_dir, mode=0o700)
        os.mkdir(output_root, 0o700)
        # Open the archive once per attempt and keep that file description for
        # every pass.  If the path is atomically replaced while this restore is
        # running, the subsequent passes still read the backup we opened first;
        # a request that opens afterwards sees the new file.  Mixing old and
        # new data in one result is therefore impossible.
        try:
            archive_file = open(archive_path, "rb")
        except FileNotFoundError as exc:
            raise PathNotFound(f"archive {archive_path!r} disappeared") from exc
        except OSError as exc:
            raise RestoreError(f"cannot open archive {archive_path!r}: {exc}") from exc
        try:
            try:
                _populate(archive_file, selection, output_root, stats, fsync)
            except (tarfile.TarError, EOFError, zlib.error) as exc:
                raise InvalidArchive(f"corrupt or truncated archive: {exc}") from exc
            stats.restored_path = _publish(
                token_dir, restore_root, restore_id, basename, fsync
            )
        finally:
            archive_file.close()
    except BaseException:
        _cleanup(
            [
                os.path.join(restore_root, restore_id),
                token_dir,
            ]
        )
        raise
    finally:
        # token_dir may be empty after a successful rename.
        if os.path.isdir(token_dir):
            _cleanup([token_dir])

    return stats
