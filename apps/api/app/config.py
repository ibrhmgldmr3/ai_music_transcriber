from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parents[1]


class Settings(BaseSettings):
    """Runtime configuration from environment variables or ``apps/api/.env``.

    Defaults don't depend on the working directory; relative paths given in the
    environment are resolved against the repository root.
    """

    model_config = SettingsConfigDict(
        env_file=API_DIR / ".env", extra="ignore", protected_namespaces=()
    )

    database_url: str = f"sqlite:///{(API_DIR / 'transcriber.db').as_posix()}"
    redis_url: str = "redis://localhost:6379/0"
    # false: run transcriptions in-process (single-process dev mode, no Redis needed);
    # true: send them to Celery workers.
    use_celery: bool = False
    storage_dir: Path = API_DIR / "storage"
    max_upload_mb: int = 50
    max_audio_minutes: float = 20.0  # longer recordings are rejected before inference
    max_notes: int = 20_000  # per transcription saved from the editor
    job_timeout_minutes: int = 60  # a pending/processing job older than this counts as stuck
    cors_origins: str = "http://localhost:3000"  # comma separated
    expose_docs: bool = True  # serve /docs and /openapi.json
    model_checkpoint: Path = REPO_ROOT / "ml" / "checkpoints" / "guitar" / "best.pt"
    # Optional second model whose tab head places notes on strings/frets.
    model_tab_checkpoint: Path | None = None
    model_device: str | None = None  # auto | cpu | cuda

    @field_validator("model_tab_checkpoint", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value  # MODEL_TAB_CHECKPOINT= disables it

    @field_validator("storage_dir", "model_checkpoint", "model_tab_checkpoint")
    @classmethod
    def _anchor_to_repo(cls, path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
