import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24_000
CHANNELS = 1

# Voices differ by several dB (ElevenLabs "rachel" is ~9 dB quieter than "adam"),
# so speech is brought to one loudness, close to macOS `say`, before output.
TARGET_RMS_DBFS = -16.0
PEAK_CEILING_DBFS = -1.0


def normalize(pcm: bytes) -> bytes:
    """Scale 16-bit PCM to TARGET_RMS_DBFS without letting the peak exceed PEAK_CEILING_DBFS."""
    if len(pcm) % 2 != 0:
        raise ValueError(f"PCM length {len(pcm)} is not a multiple of 2 (truncated data?)")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if audio.size == 0:
        return pcm
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    if rms == 0 or peak == 0:
        return pcm
    full_scale = 32768.0
    gain = min(
        full_scale * 10 ** (TARGET_RMS_DBFS / 20) / rms,
        full_scale * 10 ** (PEAK_CEILING_DBFS / 20) / peak,
    )
    return np.clip(np.round(audio * gain), -32768, 32767).astype(np.int16).tobytes()


def play_pcm(pcm: bytes) -> None:
    if len(pcm) % 2 != 0:
        raise ValueError(f"PCM length {len(pcm)} is not a multiple of 2 (truncated data?)")
    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


def write_wav(pcm: bytes, path: str | Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
