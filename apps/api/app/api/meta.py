from fastapi import APIRouter

from app.config import settings
from app.schemas import ModelInfo
from app.services.transcription import model_version, voice_version

router = APIRouter(tags=["meta"])


@router.get("/models", response_model=ModelInfo)
def current_models() -> ModelInfo:
    """The models new transcriptions use; projects with another ``model_version`` were
    transcribed by an older model (voice projects: by an older voice method)."""
    tab = settings.model_tab_checkpoint
    return ModelInfo(
        version=model_version(),
        voice_version=voice_version(),
        notes_model=settings.model_checkpoint.parent.name,
        tab_model=tab.parent.name if tab else None,
    )
