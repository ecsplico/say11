import struct
import wave
import pytest
from pathlib import Path
import numpy as np

from say_e11.audio import (
    CHANNELS,
    PEAK_CEILING_DBFS,
    SAMPLE_RATE,
    TARGET_RMS_DBFS,
    normalize,
    write_wav,
)


def test_write_wav_creates_valid_file(tmp_path):
    pcm = b"\x00\x00" * SAMPLE_RATE  # 1 second of silence
    out = tmp_path / "out.wav"
    write_wav(pcm, out)
    with wave.open(str(out)) as wf:
        assert wf.getnchannels() == CHANNELS
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnframes() == SAMPLE_RATE


def test_write_wav_preserves_content(tmp_path):
    values = [0, 100, -100, 32767, -32768]
    pcm = struct.pack(f"<{len(values)}h", *values)
    out = tmp_path / "out.wav"
    write_wav(pcm, out)
    with wave.open(str(out)) as wf:
        raw = wf.readframes(wf.getnframes())
    recovered = list(struct.unpack(f"<{len(values)}h", raw))
    assert recovered == values


def test_write_wav_accepts_path_object(tmp_path):
    pcm = b"\x00\x00" * 10
    write_wav(pcm, tmp_path / "out.wav")
    assert (tmp_path / "out.wav").exists()


def test_write_wav_accepts_string_path(tmp_path):
    pcm = b"\x00\x00" * 10
    write_wav(pcm, str(tmp_path / "out.wav"))
    assert (tmp_path / "out.wav").exists()


def _levels(pcm: bytes) -> tuple[float, float]:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(a**2)) / 32768), 20 * np.log10(np.max(np.abs(a)) / 32768)


def test_normalize_raises_quiet_speech_to_target():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    quiet = (np.sin(2 * np.pi * 220 * t) * 1000).astype(np.int16).tobytes()  # ~ -33 dBFS RMS
    rms, peak = _levels(normalize(quiet))
    assert abs(rms - TARGET_RMS_DBFS) < 0.2
    assert peak <= PEAK_CEILING_DBFS + 0.01


def test_normalize_never_pushes_peak_above_ceiling():
    a = np.zeros(SAMPLE_RATE, dtype=np.int16)
    a[100] = 8000  # one spike, almost silence otherwise: RMS target would clip
    _, peak = _levels(normalize(a.tobytes()))
    assert abs(peak - PEAK_CEILING_DBFS) < 0.01


def test_normalize_keeps_silence_unchanged():
    silence = bytes(2 * 100)
    assert normalize(silence) == silence
