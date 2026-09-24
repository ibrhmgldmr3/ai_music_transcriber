import uuid
from pathlib import Path, PurePath, PureWindowsPath

from fastapi import UploadFile

from app.config import settings

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
# Detected container -> extension the file is stored under (so it's served with the
# content type that matches what's really inside).
FORMAT_EXTENSIONS = {
    "wav": ".wav",
    "aiff": ".aiff",
    "flac": ".flac",
    "ogg": ".ogg",
    "mp3": ".mp3",
    "mp4": ".m4a",
}
MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".aiff": "audio/aiff",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}
CHUNK_SIZE = 1024 * 1024
MAX_NAME_LENGTH = 255


class InvalidUpload(ValueError):
    pass


def detect_audio_format(header: bytes) -> str | None:
    """Identify the audio container from the first bytes of a file (magic numbers)."""
    if header[:4] in (b"RIFF", b"RIFX", b"RF64") and header[8:12] == b"WAVE":
        return "wav"
    if header[:4] == b"FORM" and header[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if header[:4] == b"fLaC":
        return "flac"
    if header[:4] == b"OggS":
        return "ogg"
    if header[4:8] == b"ftyp":
        return "mp4"
    if header[:3] == b"ID3" or (len(header) > 1 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0):
        return "mp3"  # ID3 tag or MPEG frame sync
    return None


def clean_text(value: str | None, max_length: int = MAX_NAME_LENGTH) -> str:
    """Drop control characters and surrounding whitespace, then cap the length."""
    text = "".join(ch for ch in (value or "") if ch.isprintable()).strip()
    return text[:max_length]


def sanitize_filename(filename: str | None) -> str:
    """Keep only the base name (no directories, either separator), printable, <= 255 chars."""
    name = clean_text(PureWindowsPath(filename or "").name, max_length=10_000)
    if len(name) > MAX_NAME_LENGTH:
        suffix = PurePath(name).suffix[:16]
        name = name[: MAX_NAME_LENGTH - len(suffix)] + suffix
    return name


def save_upload(file: UploadFile) -> Path:
    """Stream an uploaded audio file into the storage directory under a random name."""
    suffix = PurePath(sanitize_filename(file.filename)).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise InvalidUpload(f"Unsupported file type '{suffix or '?'}'. Allowed: {allowed}")

    first = file.file.read(CHUNK_SIZE)
    if not first:
        raise InvalidUpload("File is empty")
    audio_format = detect_audio_format(first[:12])
    if audio_format is None:
        raise InvalidUpload("File content is not a supported audio format")

    storage = settings.storage_dir.resolve()
    storage.mkdir(parents=True, exist_ok=True)
    destination = storage / f"{uuid.uuid4().hex}{FORMAT_EXTENSIONS[audio_format]}"
    written, too_large = 0, False
    with destination.open("wb") as out:
        chunk = first
        while chunk:
            written += len(chunk)
            if written > settings.max_upload_bytes:
                too_large = True
                break
            out.write(chunk)
            chunk = file.file.read(CHUNK_SIZE)
    if too_large:
        destination.unlink(missing_ok=True)
        raise InvalidUpload(f"File exceeds the {settings.max_upload_mb} MB limit")
    return destination


def stored_file(path: str | Path) -> Path | None:
    """The stored file's resolved path, or None if it lies outside the storage directory."""
    candidate = Path(path).resolve()
    return candidate if candidate.is_relative_to(settings.storage_dir.resolve()) else None


def media_type(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def delete_file(path: str | Path) -> None:
    target = stored_file(path)
    if target is not None:
        target.unlink(missing_ok=True)
