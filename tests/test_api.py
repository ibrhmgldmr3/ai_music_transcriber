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
    tunings: list = []  # open strings asked for

    def transcribe(self, path, separator=None, tuning=None, progress=None):
        FakePredictor.calls.append(separator)
        FakePredictor.tunings.append(tuning)
        open_strings = list(tuning or TUNING)
        note = Note(open_strings[5], 0.1, 0.4, string=5, fret=0, confidence=0.9)
        return TranscriptionResult([note], duration=0.5, tempo=120.0, tuning=open_strings)


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
    # No beat tracker or chord recognizer by default: one tempo, as without the models.
    monkeypatch.setattr(transcription, "_recording_analysis", lambda path: None)


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


def test_notation_settings_are_saved_and_used_by_exports(client):
    import mido

    pid = upload(client, wav_bytes()).json()["id"]
    notes = [{"pitch": 65, "start": 0.0, "end": 0.5}, {"pitch": 70, "start": 0.5, "end": 1.0}]
    body = {"notes": notes, "beats_per_measure": 3, "key": "F major", "downbeat": 0.5}
    saved = client.put(f"/api/projects/{pid}/notes", json=body).json()
    assert (saved["beats_per_measure"], saved["key"], saved["downbeat"]) == (3, "F major", 0.5)
    score = client.get(f"/api/projects/{pid}/musicxml").text
    assert "<beats>3</beats>" in score and "<fifths>-1</fifths>" in score
    midi = mido.MidiFile(file=io.BytesIO(client.get(f"/api/projects/{pid}/midi").content))
    metas = {m.type: m for m in midi.tracks[0] if m.is_meta}
    assert metas["time_signature"].numerator == 3 and metas["key_signature"].key == "F"

    # Omitted fields are kept; an explicit null goes back to the estimate.
    kept = client.put(f"/api/projects/{pid}/notes", json={"notes": notes}).json()
    assert (kept["beats_per_measure"], kept["key"], kept["downbeat"]) == (3, "F major", 0.5)
    reset = client.put(
        f"/api/projects/{pid}/notes", json={"notes": notes, "key": None, "downbeat": None}
    ).json()
    assert (reset["key"], reset["downbeat"]) == (None, None)

    for bad in ({"key": "H major"}, {"beats_per_measure": 1}, {"downbeat": -1}):
        r = client.put(f"/api/projects/{pid}/notes", json={"notes": notes, **bad})
        assert r.status_code == 422, bad


def test_analysis_of_unsaved_notes(client):
    c_major, g_major = [48, 52, 55, 60], [43, 47, 50, 55]
    notes = [
        {"pitch": p, "start": beat * 0.5, "end": beat * 0.5 + 0.45}
        for beat in range(8)
        for p in (c_major if beat < 4 else g_major)
    ]
    r = client.post("/api/analysis", json={"notes": notes, "tempo": 120})
    assert r.status_code == 200, r.text
    result = r.json()
    assert [c["label"] for c in result["chords"]] == ["C", "G"]
    assert result["key"]["name"] in ("C major", "G major")
    assert result["tempo"] == 120 and result["beats_per_measure"] == 4

    chosen = client.post(
        "/api/analysis", json={"notes": notes, "tempo": 120, "key": "E minor", "downbeat": 0.5}
    ).json()
    assert chosen["key"] == {"name": "E minor", "tonic": 4, "mode": "minor", "fifths": 1}
    assert chosen["downbeat"] == 0.5 and chosen["estimated_key"] == result["key"]

    for bad in ({"tempo": 5}, {"key": "C dorian"}, {"beats_per_measure": 12}):
        assert client.post("/api/analysis", json={"notes": notes, **bad}).status_code == 422
    too_many = [notes[0]] * 51
    assert client.post("/api/analysis", json={"notes": too_many}).status_code == 422
    cross_site = client.post(
        "/api/analysis", json={"notes": notes}, headers={"Origin": "https://evil.example"}
    )
    assert cross_site.status_code == 403


def test_projects_record_their_model_and_edits(client):
    models = client.get("/api/models").json()
    assert models["version"].startswith(models["notes_model"])
    pid = upload(client, wav_bytes()).json()["id"]
    project = client.get(f"/api/projects/{pid}").json()
    assert (project["model_version"], project["edited"]) == (models["version"], False)

    notes = [{"pitch": 64, "start": 0.0, "end": 0.5}]
    client.put(f"/api/projects/{pid}/notes", json={"notes": notes})
    assert client.get(f"/api/projects/{pid}").json()["edited"] is True
    # A new transcription replaces the edits, so the flag goes back down.
    client.post(f"/api/projects/{pid}/retranscribe")
    assert client.get(f"/api/projects/{pid}").json()["edited"] is False


def test_model_version_follows_the_checkpoint_file(tmp_path, monkeypatch):
    from app.config import settings

    checkpoint = tmp_path / "guitar_vX" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(settings, "model_checkpoint", checkpoint)
    monkeypatch.setattr(settings, "model_tab_checkpoint", None)
    transcription.model_version.cache_clear()
    try:
        first = transcription.model_version()
        assert first.startswith("guitar_vX@") and first.endswith("+" + transcription.GUITAR_METHOD)
        assert first.count("+") == 1  # one checkpoint and the method
        checkpoint.write_bytes(b"retrained weights")
        transcription.model_version.cache_clear()
        assert transcription.model_version() != first
    finally:
        transcription.model_version.cache_clear()


def test_tuning_and_capo_reach_the_model_and_the_exports(client):
    r = client.post(
        "/api/projects",
        files={"file": ("take.wav", wav_bytes(), "audio/wav")},
        data={"tuning": "drop_d", "capo": "2"},
    )
    pid = r.json()["id"]
    assert (r.json()["tuning_name"], r.json()["capo"]) == ("drop_d", 2)
    assert FakePredictor.tunings[-1] == (40, 47, 52, 57, 61, 66)  # Drop D, capo 2
    transcription = client.get(f"/api/projects/{pid}/transcription").json()
    assert transcription["tuning"] == [40, 47, 52, 57, 61, 66]
    assert (transcription["tuning_name"], transcription["capo"]) == ("drop_d", 2)
    assert client.get(f"/api/projects/{pid}/tab").text.startswith("Capo 2")
    assert "<words>Capo 2</words>" in client.get(f"/api/projects/{pid}/musicxml").text

    # Changing them re-transcribes with the new strings; nonsense is refused.
    client.post(f"/api/projects/{pid}/retranscribe?tuning=standard&capo=0")
    assert FakePredictor.tunings[-1] == (40, 45, 50, 55, 59, 64)
    for bad in ("tuning=bach", "capo=13", "capo=-1"):
        assert client.post(f"/api/projects/{pid}/retranscribe?{bad}").status_code == 422
    bad_upload = client.post(
        "/api/projects", files={"file": ("t.wav", wav_bytes(), "audio/wav")}, data={"capo": "20"}
    )
    assert bad_upload.status_code == 422


def test_progress_is_visible_while_transcribing(client, monkeypatch):
    seen = {}

    class ReportingPredictor(FakePredictor):
        def transcribe(self, path, separator=None, tuning=None, progress=None):
            progress(0.4, "transcribing")
            seen.update(client.get(f"/api/projects/{holder['id']}").json())
            return super().transcribe(path, separator, tuning)

    holder = {}
    monkeypatch.setattr(transcription, "get_predictor", lambda: ReportingPredictor())
    import app.api.routes as routes

    monkeypatch.setattr(
        routes, "run_transcription", _deferred(transcription.run_transcription, holder)
    )
    pid = upload(client, wav_bytes()).json()["id"]
    holder["id"] = pid
    holder["run"]()
    assert (seen["status"], seen["progress"], seen["stage"]) == ("processing", 0.4, "transcribing")
    done = client.get(f"/api/projects/{pid}").json()
    assert (done["status"], done["progress"], done["stage"]) == ("completed", None, None)


def _deferred(run, holder):
    """Replace the background job with one the test starts once it knows the project id."""

    def schedule(project_id):
        holder["run"] = lambda: run(project_id)

    return schedule


def test_render_musicxml_of_unsaved_notes(client):
    body = {
        "notes": [{"pitch": 66, "start": 0.0, "end": 0.5, "string": 5, "fret": 0}],
        "tempo": 100,
        "tuning": [42, 47, 52, 57, 61, 66],  # standard with capo 2
        "capo": 2,
        "title": "<b>Deneme</b>",
    }
    r = client.post("/api/render/musicxml", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/vnd.recordare.musicxml+xml")
    assert "<words>Capo 2</words>" in r.text and "<b>" not in r.text
    for bad in ({"tuning": [40, 45]}, {"capo": 13}, {"tuning": [40, 45, 50, 55, 59, 300]}):
        assert client.post("/api/render/musicxml", json={**body, **bad}).status_code == 422


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
        def transcribe(self, path, separator=None, tuning=None, progress=None):
            client.delete(f"/api/projects/{holder['id']}")
            return super().transcribe(path, separator, tuning)

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


def fake_separator(y, sample_rate, stem="guitar"):
    fake_separator.stems.append(stem)
    return y


fake_separator.stems = []


def test_voice_recordings_are_transcribed_without_the_guitar_model(client):
    from ml.inference.voice import VOICE_VERSION

    FakePredictor.calls.clear()
    r = client.post(
        "/api/projects",
        files={"file": ("hum.wav", wav_bytes(seconds=1.0), "audio/wav")},
        data={"source": "voice", "tuning": "drop_d"},
    )
    assert r.status_code == 201, r.text
    project = client.get(f"/api/projects/{r.json()['id']}").json()
    assert (project["source"], project["status"]) == ("voice", "completed")
    assert project["model_version"] == VOICE_VERSION
    assert FakePredictor.calls == []  # the guitar model never ran
    transcription = client.get(f"/api/projects/{project['id']}/transcription").json()
    [note] = transcription["notes"]  # the 330 Hz tone is an E4
    assert note["pitch"] == 64 and note["string"] is not None
    assert transcription["tuning"] == [38, 45, 50, 55, 59, 64]
    assert transcription["transpose"] == 0
    assert client.get("/api/models").json()["voice_version"] == VOICE_VERSION
    curve = client.get(f"/api/projects/{project['id']}/pitch").json()
    assert curve["frame_rate"] == 50 and len(curve["values"]) == 50
    assert sum(v is not None and abs(v - 64) < 0.5 for v in curve["values"]) > 40
    # Guitar projects have none; a list of projects doesn't carry it.
    guitar = upload(client, wav_bytes()).json()["id"]
    assert client.get(f"/api/projects/{guitar}/pitch").status_code == 404
    assert all("pitch_curve" not in p for p in client.get("/api/projects").json())


def test_retranscribe_can_switch_between_guitar_and_voice(client, monkeypatch):
    monkeypatch.setattr(transcription, "get_separator", lambda: fake_separator)
    pid = upload(client, wav_bytes()).json()["id"]
    FakePredictor.calls.clear()
    fake_separator.stems.clear()
    voice = client.post(f"/api/projects/{pid}/retranscribe?source=voice&separate_guitar=true")
    assert voice.json()["source"] == "voice"
    assert fake_separator.stems == ["vocals"]  # song mode isolates the singer
    assert client.post(f"/api/projects/{pid}/retranscribe?source=guitar").json()["source"] == (
        "guitar"
    )
    assert FakePredictor.calls == [fake_separator]  # the guitar model gets the guitar stem
    assert client.post(f"/api/projects/{pid}/retranscribe?source=drums").status_code == 422
    bad = client.post(
        "/api/projects",
        files={"file": ("take.wav", wav_bytes(), "audio/wav")},
        data={"source": "piano"},
    )
    assert bad.status_code == 422


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
    added = (
        "separate_guitar",
        "beats_per_measure",
        "key_name",
        "downbeat",
        "model_version",
        "edited",
        "tuning_name",
        "capo",
        "progress",
        "stage",
        "source",
        "transpose",
        "pitch_curve",
    )
    with engine.begin() as connection:  # simulate a database from the first release
        for column in added:
            connection.execute(text(f"ALTER TABLE projects DROP COLUMN {column}"))
    assert not set(added) & {c["name"] for c in inspect(engine).get_columns("projects")}
    init_db()
    project = client.get(f"/api/projects/{pid}").json()
    assert (project["separate_guitar"], project["source"]) == (False, "guitar")
    transcription = client.get(f"/api/projects/{pid}/transcription").json()
    assert (transcription["beats_per_measure"], transcription["key"]) == (4, None)
    assert transcription["transpose"] == 0


def fake_song(monkeypatch, capo_seen: list):
    import ml.inference.song as song
    from music_core.tab import open_strings

    def transcribe(path, separator, chords, beats, tuning_name="standard", capo=None, **kw):
        capo_seen.append(capo)
        capo = 3 if capo is None else capo
        melody = [Note(67, 0.5, 1.0, string=3, fret=9 - capo)]
        return song.SongResult(
            notes=melody,
            duration=4.0,
            tempo=120.0,
            tuning=list(open_strings(tuning_name, capo)),
            chords=[
                {"start": 0.0, "end": 2.0, "label": "C:min"},
                {"start": 2.0, "end": 4.0, "label": "A#:maj"},
            ],
            beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            downbeats=[0.0, 2.0],
            capo=capo,
            beats_per_measure=4,
            downbeat=0.0,
            key="F major",
        )

    monkeypatch.setattr(song, "transcribe_song", transcribe)
    monkeypatch.setattr(transcription, "_load_song_models", lambda: (None, None, None))


def test_songs_get_chords_bars_and_an_automatic_capo(client, monkeypatch):
    from ml.inference.song import SONG_VERSION
    from ml.inference.voice import VOICE_VERSION

    capo_seen: list = []
    fake_song(monkeypatch, capo_seen)
    r = client.post(
        "/api/projects",
        files={"file": ("song.wav", wav_bytes(), "audio/wav")},
        data={"source": "song", "capo": "-1"},
    )
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    project = client.get(f"/api/projects/{pid}").json()
    assert (project["source"], project["capo"], project["capo_auto"]) == ("song", 3, True)
    assert project["model_version"] == f"{SONG_VERSION}+{VOICE_VERSION}" and capo_seen == [None]
    assert client.get("/api/models").json()["song_version"] == project["model_version"]
    t = client.get(f"/api/projects/{pid}/transcription").json()
    assert [c["label"] for c in t["chords"]] == ["C:min", "A#:maj"]
    assert t["downbeats"] == [0.0, 2.0] and len(t["beats"]) == 8
    assert (t["key"], t["capo"], t["tuning"]) == ("F major", 3, [43, 48, 53, 58, 62, 67])

    sheet = client.get(f"/api/projects/{pid}/chordsheet").text
    assert "Ton: F major · Capo 3" in sheet and "| Am " in sheet and "(= Cm)" in sheet
    xml = client.get(f"/api/projects/{pid}/musicxml").text
    assert '<kind text="m">minor</kind>' in xml and "<root-step>B</root-step>" in xml

    # Chords are edited and saved with the notes; nonsense is refused.
    edited = [{"start": 0.0, "end": 4.0, "label": "F:maj7"}]
    r = client.put(f"/api/projects/{pid}/notes", json={"notes": t["notes"], "chords": edited})
    assert r.status_code == 200 and r.json()["chords"] == edited
    bad = {"notes": t["notes"], "chords": [{"start": 0, "end": 1, "label": "H:maj"}]}
    assert client.put(f"/api/projects/{pid}/notes", json=bad).status_code == 422

    # A chosen capo is kept; an automatic one only makes sense for songs.
    client.post(f"/api/projects/{pid}/retranscribe?capo=0")
    assert capo_seen[-1] == 0 and client.get(f"/api/projects/{pid}").json()["capo_auto"] is False
    guitar = upload(client, wav_bytes()).json()["id"]
    assert client.post(f"/api/projects/{guitar}/retranscribe?capo=-1").status_code == 422
    assert client.get(f"/api/projects/{guitar}/chordsheet").status_code == 404


def test_chord_voicings_name_and_finger_shapes_from_a_capo(client):
    r = client.post(
        "/api/chords/voicings",
        json={"labels": ["C:min", "A#", "N"], "capo": 3, "key": "F major"},
    )
    assert r.status_code == 200
    cm, bb, none = r.json()
    assert (cm["name"], cm["shape"], cm["frets"]) == ("Cm", "Am", [-1, 0, 2, 2, 1, 0])
    assert (bb["name"], bb["shape"]) == ("Bb", "G")
    assert none["frets"] is None
    bad = client.post("/api/chords/voicings", json={"labels": ["Q"]})
    assert bad.status_code == 422


def drifting_beats(count: int = 24, start: float = 0.3) -> list[float]:
    """A player without a click: beats 0.5 s apart, slowing down to 0.6 s."""
    beats, t = [], start
    for i in range(count):
        beats.append(round(t, 4))
        t += 0.5 + 0.1 * i / count
    return beats


def test_guitar_projects_follow_the_tracked_beats(client, monkeypatch):
    beats = drifting_beats()
    recording = {
        "beats": beats,
        "downbeats": beats[1::3],  # 3/4, the bar starting on the second beat
        "chord_scores": [{"start": 0.0, "top": [["E:min", -0.1], ["G:maj", -2.0]]}],
    }
    monkeypatch.setattr(transcription, "_recording_analysis", lambda path: recording)
    pid = upload(client, wav_bytes()).json()["id"]
    t = client.get(f"/api/projects/{pid}/transcription").json()
    assert t["beats"] == beats and t["beat_grid"] is True
    assert t["beats_per_measure"] == 3 and t["downbeat"] is None
    assert t["tempo"] == pytest.approx(60 / np.median(np.diff(beats)), abs=0.01)

    request = {"notes": t["notes"], "tempo": t["tempo"], "beats_per_measure": 3, "project_id": pid}
    analysis = client.post("/api/analysis", json=request).json()
    assert analysis["tracked"] is True
    assert all(any(abs(b - x) < 1e-3 for x in analysis["beats"]) for b in beats)
    assert [b["time"] for b in analysis["bars"]][:3] == pytest.approx(beats[1:8:3], abs=1e-3)
    # The note at 0.1 s is in bar 1, which starts before the recording: the first bar
    # line in it is bar 2's, as the MusicXML export numbers them.
    assert analysis["bars"][0]["number"] == 2
    fixed = client.post(
        "/api/analysis", json={**request, "beat_grid": False, "duration": 10}
    ).json()
    assert fixed["tracked"] is False and len(set(np.round(np.diff(fixed["beats"]), 3))) == 1
    assert client.post("/api/analysis", json={**request, "project_id": "nope"}).status_code == 404

    # The exports follow the drift: tempo changes in MusicXML, a tempo map in MIDI.
    played = [{"pitch": 64, "start": b, "end": b + 0.3, "string": 5, "fret": 0} for b in beats]
    client.put(f"/api/projects/{pid}/notes", json={"notes": played})
    xml = client.get(f"/api/projects/{pid}/musicxml").text
    assert xml.count("<sound tempo=") > 1
    import mido

    midi = mido.MidiFile(file=io.BytesIO(client.get(f"/api/projects/{pid}/midi").content))
    assert sum(m.type == "set_tempo" for m in midi.tracks[0]) > 5

    # Switching the grid off is saved like the rest of the notation.
    r = client.put(f"/api/projects/{pid}/notes", json={"notes": played, "beat_grid": False})
    assert r.json()["beat_grid"] is False
    midi = mido.MidiFile(file=io.BytesIO(client.get(f"/api/projects/{pid}/midi").content))
    assert sum(m.type == "set_tempo" for m in midi.tracks[0]) == 1


def test_songs_get_key_changes_slash_chords_and_a_strumming_pattern(client, monkeypatch):
    import ml.inference.song as song

    capo_seen: list = []
    fake_song(monkeypatch, capo_seen)
    plain = song.transcribe_song

    def modulating(*args, **kwargs):
        result = plain(*args, **kwargs)
        result.chords = [
            {"start": 0.0, "end": 2.0, "label": "D:maj/3"},
            {"start": 2.0, "end": 4.0, "label": "A#:maj"},
        ]
        result.keys = [
            {"start": 0.0, "end": 2.0, "key": "D major"},
            {"start": 2.0, "end": 4.0, "key": "F major"},
        ]
        result.strums = [(0.0, 3.0), (0.5, 3.0), (0.75, 3.0), (1.0, 3.0), (1.5, 3.0)] * 1 + [
            (2.0, 3.0), (2.5, 3.0), (2.75, 3.0), (3.0, 3.0), (3.5, 3.0)
        ]  # fmt: skip
        result.key = "F major"
        return result

    monkeypatch.setattr(song, "transcribe_song", modulating)
    r = client.post(
        "/api/projects",
        files={"file": ("song.wav", wav_bytes(), "audio/wav")},
        data={"source": "song", "capo": "0"},
    )
    pid = r.json()["id"]
    t = client.get(f"/api/projects/{pid}/transcription").json()
    assert t["key"] is None and [k["key"] for k in t["keys"]] == ["D major", "F major"]
    assert t["chords"][0]["label"] == "D:maj/3"

    request = {"notes": t["notes"], "tempo": 120, "project_id": pid}
    analysis = client.post("/api/analysis", json=request).json()
    assert [k["key"] for k in analysis["keys"]] == ["D major", "F major"]
    assert analysis["rhythm"]["text"] == "D-DUD-D-"  # eighths: 1, 2 &, 3 (and 4)
    assert analysis["rhythm"]["per_beat"] == 2

    sheet = client.get(f"/api/projects/{pid}/chordsheet").text
    assert "D/F#" in sheet and "200232" in sheet  # the bass on the low string
    assert "Ton değişimi: ölçü 2: F major" in sheet and "Ritim: D-DUD-D-" in sheet
    xml = client.get(f"/api/projects/{pid}/musicxml").text
    assert "<bass-step>F</bass-step>" in xml and xml.count("<fifths>") == 2
