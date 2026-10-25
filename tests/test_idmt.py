from ml.datasets.idmt import find_takes, load_annotation

XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<instrumentRecording>
  <transcription>
    <event>
      <onsetSec>1.0222</onsetSec><pitch>60</pitch><offsetSec>1.2572</offsetSec>
      <excitationStyle>FS</excitationStyle><expressionStyle>SL</expressionStyle>
      <fretNumber>5</fretNumber><stringNumber>4</stringNumber>
      <modulationFrequency>2.128</modulationFrequency>
      <modulationFrequencyRange>-200</modulationFrequencyRange>
    </event>
    <event>
      <onsetSec>0.5</onsetSec><pitch>40</pitch><offsetSec>0.9</offsetSec>
      <excitationStyle>PK</excitationStyle><expressionStyle>BE</expressionStyle>
      <fretNumber>0</fretNumber><stringNumber>1</stringNumber>
      <modulationFrequency>0</modulationFrequency>
      <modulationFrequencyRange>0</modulationFrequencyRange>
    </event>
  </transcription>
</instrumentRecording>
"""


def test_annotation_strings_count_from_the_low_e(tmp_path):
    path = tmp_path / "AR_test.xml"
    path.write_text(XML, encoding="utf-8")
    bend, slide = load_annotation(path)  # sorted by onset
    assert (bend.note.pitch, bend.note.string, bend.note.fret, bend.expression) == (40, 0, 0, "BE")
    assert bend.slide_cents is None
    assert (slide.note.string, slide.note.fret, slide.expression) == (3, 5, "SL")
    assert slide.slide_cents == -200 and slide.excitation == "FS"


def test_lick_11_annotations_find_their_audio(tmp_path):
    root = tmp_path / "dataset2"
    (root / "annotation").mkdir(parents=True)
    (root / "audio").mkdir()
    for stem in ("AR_Lick11_FNVSBHD", "AR_Lick1_FVSDN", "LP_Lick2_KN"):
        (root / "annotation" / f"{stem}.xml").write_text(XML, encoding="utf-8")
    for stem in ("AR_Lick11_FN", "AR_Lick1_FN", "AR_Lick1_FVSDN", "LP_Lick2_KN"):
        (root / "audio" / f"{stem}.wav").write_bytes(b"")
    takes = {t.name: t.audio_path.name for t in find_takes(tmp_path, guitars=("AR",))}
    assert takes == {
        "AR_Lick11_FNVSBHD": "AR_Lick11_FN.wav",
        "AR_Lick1_FVSDN": "AR_Lick1_FVSDN.wav",
    }
