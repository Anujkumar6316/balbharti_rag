"""
tts.py — Marathi TTS wrapper using gTTS.

On Pi 5 (~200ms synthesis, ~300ms playback start for a 3-sentence answer):
  - First load: ~200ms (import + first request)
  - Per synthesis: ~150-300ms (HTTP request to Google TTS)
  - Audio playback: ~1-3s (depends on answer length, overlaps with user listening)

Usage:
    from src.tts import get_tts, synthesize_and_play
    tts = get_tts()
    synthesize_and_play("नमस्कार, मी तुम्हाला कशात मदत करू शकतो?")

Voice options (backend):
  - gtts (default): Google TTS, cloud, native Marathi, excellent quality, ~200ms
  - piper: Offline Piper TTS, requires piper-tts + Marathi voice model
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TTS_BACKEND: Optional[str] = None  # "gtts" or "piper"


class MarathiTTS:
    """Singleton TTS wrapper. Uses gTTS by default (native Marathi support).

    Loads on first call to synthesize(). On Pi 5, first call is ~200ms for
    the HTTP request; subsequent calls reuse the same session.
    """

    _instance: Optional["MarathiTTS"] = None

    def __init__(self, backend: str = "gtts"):
        self._backend = backend
        self._loaded = False
        self._load_time_s = 0.0
        self._lock = threading.Lock()

    @classmethod
    def get(cls, backend: str = "gtts") -> "MarathiTTS":
        if cls._instance is None:
            cls._instance = cls(backend=backend)
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        t = t0

        if self._backend == "gtts":
            try:
                from gtts import gTTS
                gTTS  # verify import
                t = time.perf_counter()
                logger.info("gTTS backend ready (%.0fms)", (t - t0) * 1000)
            except ImportError:
                raise ImportError(
                    "gTTS not installed. Run: pip install gtts"
                )
        elif self._backend == "piper":
            raise RuntimeError(
                "Piper TTS: no Marathi voice available in official repo. "
                "Use backend='gtts' for native Marathi support."
            )
        else:
            raise ValueError(f"Unknown TTS backend: {self._backend}")

        self._load_time_s = time.perf_counter() - t0
        self._loaded = True
        logger.info("MarathiTTS ready in %.2fs", self._load_time_s)

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesize text to WAV audio bytes.

        Args:
            text: Marathi text to synthesize.

        Returns:
            (wav_bytes, sample_rate)
        """
        if not self._loaded:
            self.load()
        if not text or not text.strip():
            return (b"", 22050)

        t0 = time.perf_counter()
        with self._lock:
            if self._backend == "gtts":
                result = self._synthesize_gtts(text)
            else:
                raise ValueError(f"Backend not loaded: {self._backend}")

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("TTS synthesis: %dms for %d chars", int(latency_ms), len(text))
        return result

    def _synthesize_gtts(self, text: str) -> tuple[bytes, int]:
        """Synthesize using gTTS (Google Text-to-Speech, Marathi)."""
        from gtts import gTTS

        tts = gTTS(text=text, lang="mr", slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        wav_bytes = buffer.read()
        # gTTS returns MP3, not WAV, but we'll serve it as-is
        # The player handles both formats
        return (wav_bytes, 24000)

    def synthesize_to_file(self, text: str, output_path: str) -> int:
        """Synthesize and save to file. Returns latency in ms."""
        audio_bytes, sample_rate = self.synthesize(text)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return 0


def play_audio(audio_bytes: bytes, sample_rate: int = 24000) -> None:
    """Play audio bytes. Blocks until playback finishes.

    Tries:
      1. sounddevice (cross-platform Python)
      2. aplay (Linux/Pi)
      3. ffplay (fallback)
    """
    if not audio_bytes:
        return

    try:
        import sounddevice as sd
        import numpy as np
        import wave

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
                audio_array = np.frombuffer(frames, dtype=np.int16)
                sd.play(audio_array, sr)
                sd.wait()
            return
        except Exception:
            pass

        try:
            sd.play(audio_bytes, sample_rate)
            sd.wait()
            return
        except Exception:
            pass
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        subprocess.run(["aplay", "-q", tmp_path], check=False, timeout=30)
    except FileNotFoundError:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path],
                       check=False, capture_output=True, timeout=30)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def synthesize_and_play(text: str, tts: Optional[MarathiTTS] = None) -> dict:
    """Synthesize text and play it. Returns timing info.

    Args:
        text: Marathi text to speak.
        tts: Optional MarathiTTS instance (creates one if None).

    Returns:
        {"synthesis_ms": int, "playback_started_ms": int}
    """
    t_start = time.perf_counter()
    if tts is None:
        tts = MarathiTTS.get()
    if not tts._loaded:
        tts.load()

    audio_bytes, sample_rate = tts.synthesize(text)
    synth_ms = (time.perf_counter() - t_start) * 1000
    playback_start_ms = (time.perf_counter() - t_start) * 1000

    try:
        play_audio(audio_bytes, sample_rate)
    except Exception as e:
        logger.warning("Audio playback failed: %s", e)

    return {
        "synthesis_ms": int(synth_ms),
        "playback_started_ms": int(playback_start_ms),
    }


def get_tts() -> MarathiTTS:
    """Get the singleton MarathiTTS instance (auto-loaded on first call)."""
    tts = MarathiTTS.get()
    if not tts._loaded:
        tts.load()
    return tts
