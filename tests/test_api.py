"""API security and robustness tests (a fake predictor stands in for the model)."""

import io
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sf = pytest.importorskip("soundfile")

# Settings are read at import time, so configure the environment first.
_TMP = Path(tempfile.mkdtemp(prefix="api-test-"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    STORAGE_DIR=str(_TMP / "storage"),
    USE_CELERY="false",
    MAX_UPLOAD_MB="1",
    MAX_AUDIO_MINUTES="1",
    MAX_NOTES="50",
    CORS_ORIGINS="http://localhost:3000",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from fastapi.testclient import TestClient  # noqa: E402

import app.services.transcription as transcription  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Project, ProjectStatus, SessionLocal  # noqa: E402
from app.services.storage import detect_audio_format, sanitize_filename  # noqa: E402
from ml.inference.predict import TranscriptionResult  # noqa: E402
from music_core.notes import Note  # noqa: E402

TUNING = [40, 45, 50, 55, 59, 64]


class FakePredictor:
    calls: list = []  # separator passed to each transcription

    def transcribe(self, path, separator=None):
        FakePredictor.calls.append(separator)
        note = Note(64, 0.1, 0.4, string=5, fret=0, confidence=0.9)
        return TranscriptionResult([note], duration=0.5, tempo=120.0, tuning=TUNING)


def wav_bytes(seconds: float = 0.5, sr: int = 8000) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 330 * t), sr, format="WAV")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(transcription, "get_predictor", lambda: FakePredictor())


def upload(client, data: bytes, filename: str = "take.wav", **kwargs):
    return client.post("/api/projects", files={"file": (filename, data, "audio/wav")}, **kwargs)


def test_upload_transcribes_and_exports(client):
    r = upload(client, wav_bytes())
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["status"] == "completed"
    assert len(client.get(f"/api/projects/{pid}/transcription").json()["notes"]) == 1
    audio = client.get(f"/api/projects/{pid}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.headers["x-content-type-options"] == "nosniff"
    assert client.get(f"/api/projects/{pid}/midi").status_code == 200
    assert client.get(f"/api/projects/{pid}/tab").status_code == 200
    score = client.get(f"/api/projects/{pid}/musicxml")
    assert score.status_code == 200
    assert score.headers["content-type"] == "application/vnd.recordare.musicxml+xml"
    assert b"<score-partwise" in score.content


def test_non_audio_content_is_rejected(client):
    r = upload(client, b"<html><script>alert(1)</script></html>", "evil.wav")
    assert r.status_code == 400
    assert "not a supported audio format" in r.json()["detail"]


def test_disallowed_extension_is_rejected(client):
    assert upload(client, wav_bytes(), "take.exe").status_code == 400


def test_empty_upload_is_rejected(client):
    assert upload(client, b"").status_code == 400


def test_oversized_upload_is_cut_off(client):
    r = upload(client, wav_bytes() + b"\0" * (3 * 1024 * 1024))
    assert r.status_code == 413


def test_filename_is_sanitized(client):
    r = upload(client, wav_bytes(), "..\\..\\" + "a" * 300 + ".wav")
    assert r.status_code == 201
    filename = r.json()["filename"]
    assert len(filename) <= 255 and filename.endswith(".wav")
    assert "/" not in filename and "\\" not in filename and ".." not in filename


@pytest.mark.parametrize("method", ["post", "delete"])
def test_cross_site_requests_are_blocked(client, method):
    headers = {"Origin": "https://evil.example"}
    if method == "post":
        r = upload(client, wav_bytes(), headers=headers)
    else:
        pid = upload(client, wav_bytes()).json()["id"]
        r = client.delete(f"/api/projects/{pid}", headers=headers)
    assert r.status_code == 403


def test_allowed_origin_and_no_origin_pass(client):
    assert (
        upload(client, wav_bytes(), headers={"Origin": "http://localhost:3000"}).status_code == 201
    )
    assert upload(client, wav_bytes()).status_code == 201


@pytest.mark.parametrize(
    "note",
    [
        {"pitch": 60, "start": float("nan"), "end": 1.0},
        {"pitch": 60, "start": 1e7, "end": 1e7 + 1},
        {"pitch": 60, "start": 1.0, "end": 0.5},
        {"pitch": 60, "start": 0.0, "end": 1.0, "string": 2},
        {"pitch": 128, "start": 0.0, "end": 1.0},
    ],
)
def test_invalid_notes_are_rejected(client, note):
    pid = upload(client, wav_bytes()).json()["id"]
    r = client.put(
        f"/api/projects/{pid}/notes",
        content=json.dumps({"notes": [note]}),  # json.dumps emits NaN literally
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    assert client.get(f"/api/projects/{pid}/transcription").status_code == 200


def test_saving_assigns_missing_positions_and_keeps_chosen_ones(client):
    pid = upload(client, wav_bytes()).json()["id"]
    notes = [
        {"pitch": 64, "start": 0.0, "end": 0.5},  # no position -> optimizer picks one
        {"pitch": 69, "start": 1.0, "end": 1.5, "string": 1, "fret": 24},  # user's choice
        {"pitch": 30, "start": 2.0, "end": 2.5},  # below the guitar's range
    ]
    saved = client.put(f"/api/projects/{pid}/notes", json={"notes": notes}).json()["notes"]
    placed, chosen, unplayable = saved
    assert placed["string"] is not None and TUNING[placed["string"]] + placed["fret"] == 64
    assert (chosen["string"], chosen["fret"]) == (1, 24)
    assert unplayable["string"] is None and unplayable["fret"] is None


def test_corrected_tempo_is_saved_and_used_by_exports(client):
    pid = upload(client, wav_bytes()).json()["id"]
    notes = [{"pitch": 64, "start": 0.0, "end": 0.5}]
    saved = client.put(f"/api/projects/{pid}/notes", json={"notes": notes, "tempo": 97})
    assert saved.json()["tempo"] == 97
    score = client.get(f"/api/projects/{pid}/musicxml").text
    assert "<per-minute>97</per-minute>" in score
    # Saving without a tempo keeps the current one; absurd values are rejected.
    assert client.put(f"/api/projects/{pid}/notes", json={"notes": notes}).json()["tempo"] == 97
    for bad in (0, 1000):
        r = client.put(f"/api/projects/{pid}/notes", json={"notes": notes, "tempo": bad})
        assert r.status_code == 422


def test_note_count_is_limited(client):
    pid = upload(client, wav_bytes()).json()["id"]
    notes = [{"pitch": 60, "start": 0.0, "end": 1.0}] * 51
    assert client.put(f"/api/projects/{pid}/notes", json={"notes": notes}).status_code == 422


def test_too_long_recording_fails_with_clear_message(client):
    r = upload(client, wav_bytes(seconds=61, sr=2000))
    project = client.get(f"/api/projects/{r.json()['id']}").json()
    assert project["status"] == "failed"
    assert "limit is 1 minutes" in project["error"]


def test_internal_errors_are_not_leaked(client, monkeypatch):
    def broken():
        raise RuntimeError("secret detail at /srv/private/model.pt")

    monkeypatch.setattr(transcription, "get_predictor", broken)
    project = client.get(f"/api/projects/{upload(client, wav_bytes()).json()['id']}").json()
    assert project["status"] == "failed"
    assert "secret" not in project["error"] and "/srv" not in project["error"]


def _set_status(pid: str, status: ProjectStatus, age: timedelta) -> None:
    with SessionLocal() as db:
        project = db.get(Project, pid)
        project.status = status
        db.commit()
        project.updated_at = datetime.now(timezone.utc) - age  # after onupdate fired
        db.commit()


def test_stuck_job_can_be_restarted_only_after_timeout(client):
    pid = upload(client, wav_bytes()).json()["id"]
    _set_status(pid, ProjectStatus.processing, timedelta(minutes=5))
    assert client.post(f"/api/projects/{pid}/retranscribe").status_code == 409
    _set_status(pid, ProjectStatus.processing, timedelta(hours=2))
    assert client.post(f"/api/projects/{pid}/retranscribe").status_code == 200
    assert client.get(f"/api/projects/{pid}").json()["status"] == "completed"


def test_deleted_project_during_transcription_is_skipped(client, monkeypatch):
    class DeletingPredictor(FakePredictor):
        def transcribe(self, path, separator=None):
            client.delete(f"/api/projects/{holder['id']}")
            return super().transcribe(path, separator)

    holder = {}
    pid = upload(client, wav_bytes()).json()["id"]
    holder["id"] = pid
    monkeypatch.setattr(transcription, "get_predictor", lambda: DeletingPredictor())
    transcription.run_transcription(pid)  # must not raise
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_list_is_paginated(client):
    for _ in range(3):
        upload(client, wav_bytes())
    assert len(client.get("/api/projects?limit=2").json()) == 2
    assert client.get("/api/projects?limit=100000").status_code == 422


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in r.headers["content-security-policy"]


def test_timestamps_carry_utc_offset(client):
    # Without an offset, browsers read the (UTC) value as local time.
    project = upload(client, wav_bytes()).json()
    for field in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(project[field].replace("Z", "+00:00"))
        assert parsed.utcoffset() == timedelta(0)
        assert abs(datetime.now(timezone.utc) - parsed) < timedelta(minutes=1)


def test_audio_varies_on_origin(client):
    # The <audio> element fetches without Origin, the waveform with CORS; the cached
    # response must not be shared between the two.
    pid = upload(client, wav_bytes()).json()["id"]
    plain = client.get(f"/api/projects/{pid}/audio")
    cors = client.get(f"/api/projects/{pid}/audio", headers={"Origin": "http://localhost:3000"})
    assert "origin" in plain.headers["vary"].lower()
    assert "origin" in cors.headers["vary"].lower()
    assert cors.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_detect_audio_format():
    assert detect_audio_format(wav_bytes()[:12]) == "wav"
    assert detect_audio_format(b"fLaC\0\0\0\0\0\0\0\0") == "flac"
    assert detect_audio_format(b"ID3\x04\0\0\0\0\0\0\0\0") == "mp3"
    assert detect_audio_format(b"\0\0\0\x20ftypM4A ") == "mp4"
    assert detect_audio_format(b"<!DOCTYPE html>") is None


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\x\\song.wav") == "song.wav"
    assert sanitize_filename("a\x00b\nc.wav") == "abc.wav"


def fake_separator(y, sample_rate):
    return y


def test_song_mode_separates_the_guitar_first(client, monkeypatch):
    monkeypatch.setattr(transcription, "get_separator", lambda: fake_separator)
    FakePredictor.calls.clear()
    solo = upload(client, wav_bytes()).json()
    song = client.post(
        "/api/projects",
        files={"file": ("band.wav", wav_bytes(), "audio/wav")},
        data={"separate_guitar": "true"},
    ).json()
    assert (solo["separate_guitar"], song["separate_guitar"]) == (False, True)
    assert FakePredictor.calls == [None, fake_separator]
    assert client.get(f"/api/projects/{song['id']}").json()["status"] == "completed"


def test_retranscribe_can_switch_song_mode(client, monkeypatch):
    monkeypatch.setattr(transcription, "get_separator", lambda: fake_separator)
    pid = upload(client, wav_bytes()).json()["id"]
    FakePredictor.calls.clear()
    on = client.post(f"/api/projects/{pid}/retranscribe?separate_guitar=true").json()
    assert on["separate_guitar"] is True
    kept = client.post(f"/api/projects/{pid}/retranscribe").json()  # unchanged when omitted
    assert kept["separate_guitar"] is True
    off = client.post(f"/api/projects/{pid}/retranscribe?separate_guitar=false").json()
    assert off["separate_guitar"] is False
    assert FakePredictor.calls == [fake_separator, fake_separator, None]


def test_song_mode_without_demucs_fails_clearly(client, monkeypatch):
    def missing():
        raise ImportError("No module named 'demucs'")

    monkeypatch.setattr(transcription, "get_separator", missing)
    r = client.post(
        "/api/projects",
        files={"file": ("band.wav", wav_bytes(), "audio/wav")},
        data={"separate_guitar": "true"},
    )
    project = client.get(f"/api/projects/{r.json()['id']}").json()
    assert project["status"] == "failed" and "demucs" in project["error"]


def test_existing_database_gets_new_columns(client):
    from sqlalchemy import inspect, text

    from app.models.database import engine, init_db

    pid = upload(client, wav_bytes()).json()["id"]
    with engine.begin() as connection:  # simulate a database from before song mode
        connection.execute(text("ALTER TABLE projects DROP COLUMN separate_guitar"))
    assert "separate_guitar" not in {c["name"] for c in inspect(engine).get_columns("projects")}
    init_db()
    assert client.get(f"/api/projects/{pid}").json()["separate_guitar"] is False
