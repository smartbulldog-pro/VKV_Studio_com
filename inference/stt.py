"""
Synapse STT — Speech-to-Text (Google-native, Gemma-4 audio)
===========================================================
Single backend now that the stack is all-Google:

- GemmaSTT: Gemma-4 E4B native audio (multimodal ASR). Folds STT into the E4B
  model itself — no separate weights/process. STUB for now: transcribe() raises
  NotImplementedError until the audio-mmproj GGUF is exported (the DSP preprocessing
  IS implemented and testable). See config.STT_BACKEND + inference/STT_BACKEND.md.

Whisper (faster-whisper) was REMOVED 2026-07-06 — consolidating onto Google, same
call as dropping Coqui XTTS. Until the mmproj GGUF ships, voice INPUT is offline
(an accepted trade); text chat and TTS voice OUTPUT are unaffected.

The backend satisfies the contract:
    transcribe(audio_bytes: bytes) -> TranscriptionResult
    is_loaded -> bool
    unload() -> None
"""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    GEMMA_STT_CHUNK_OVERLAP_SEC,
    GEMMA_STT_CTX_SIZE,
    GEMMA_STT_GPU_LAYERS,
    GEMMA_STT_HOST,
    GEMMA_STT_MAX_DURATION_SEC,
    GEMMA_STT_MAX_SEGMENT_SEC,
    GEMMA_STT_MMPROJ_PATH,
    GEMMA_STT_MODEL_NAME,
    GEMMA_STT_MODEL_PATH,
    GEMMA_STT_PORT,
    GEMMA_STT_PROMPT,
    GEMMA_STT_SAMPLE_RATE,
    GEMMA_STT_SERVER_URL,
    GEMMA_STT_STARTUP_TIMEOUT,
    GEMMA_STT_TIMEOUT,
    GOOGLE_STT_ALT_LANGUAGES,
    GOOGLE_STT_BUDGET_FILE,
    GOOGLE_STT_BUDGET_SECONDS,
    GOOGLE_STT_CREDENTIALS,
    GOOGLE_STT_ENDPOINT,
    GOOGLE_STT_LANGUAGE,
    GOOGLE_STT_MAX_BYTES,
    GOOGLE_STT_MAX_SECONDS,
    GOOGLE_STT_MODEL,
    GOOGLE_STT_TIMEOUT,
    STT_BACKEND,
)
# Reuse the LLM router's llama-server binary + DLL-path setup (imported at module
# load in llm.py). The GemmaSTT audio child inherits that environment.
from llm import LLAMA_BIN_DIR, LLAMA_SERVER_EXE
from log_safety import child_env, safe_err as _safe_err

logger = logging.getLogger("synapse.stt")


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of a speech-to-text transcription."""
    text: str
    language: str          # ISO 639-1 code, e.g. "ru", "en"
    language_prob: float   # Confidence of language detection (0.0-1.0)
    duration: float        # Audio duration in seconds


@dataclass(frozen=True)
class AudioChunk:
    """A ≤30 s slice of 16 kHz mono float32 [-1,1] audio, ready for Gemma audio-in."""
    samples: "object"      # numpy.ndarray float32, shape (n,) — annotated loosely to avoid a numpy import at module load
    start_sec: float
    end_sec: float


# ── Gemma audio preprocessing (pure signal DSP — no model, fully unit-testable) ──
# These module-level functions implement Gemma's audio input contract
# (16 kHz mono float32 [-1,1], ≤30 s chunks). They take/return numpy arrays and do
# NOT touch the model, GPU, or network, so they can be tested with synthetic signals.


def to_mono(samples: "object") -> "object":
    """Downmix to a single channel.

    Accepts a 1-D array (already mono) or a 2-D array shaped (channels, n) — the
    layout PyAV yields for planar float audio — and averages across channels.
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0).astype(np.float32)
    raise ValueError(f"to_mono expects 1-D or 2-D audio, got ndim={arr.ndim}")


def resample_linear(samples: "object", src_sr: int, dst_sr: int) -> "object":
    """Resample a 1-D signal from src_sr to dst_sr via linear interpolation.

    No anti-alias filter — adequate for a preprocessing baseline (the model's mel
    front-end is tolerant), and dependency-free (pure numpy). Deterministic length:
    round(n * dst_sr / src_sr).
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0 or src_sr == dst_sr:
        return arr
    n_src = arr.shape[-1]
    n_dst = int(round(n_src * dst_sr / src_sr))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    # Map each destination index back onto the source time axis and interpolate.
    dst_positions = np.arange(n_dst, dtype=np.float64) * (src_sr / dst_sr)
    src_index = np.arange(n_src, dtype=np.float64)
    return np.interp(dst_positions, src_index, arr).astype(np.float32)


def normalize_peak(samples: "object", target_peak: float = 0.98) -> "object":
    """Guarantee the signal sits within [-1, 1].

    Scales down any signal whose peak exceeds 1.0 (to target_peak) and clips as a
    safety net. Quiet audio is left untouched — we don't amplify it, to avoid
    blowing up background noise.
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0 and peak > 1e-9:
        arr = arr * (target_peak / peak)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def chunk_audio(
    samples: "object",
    sample_rate: int = GEMMA_STT_SAMPLE_RATE,
    max_sec: float = GEMMA_STT_MAX_SEGMENT_SEC,
    overlap_sec: float = GEMMA_STT_CHUNK_OVERLAP_SEC,
) -> "list[AudioChunk]":
    """Split a 1-D signal into ≤max_sec chunks with a small overlap.

    Every returned chunk is guaranteed ≤ max_sec (Gemma's hard 30 s limit). Chunks
    after the first begin `overlap_sec` before the previous chunk's end so a word
    straddling the boundary survives intact in at least one chunk.
    """
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    n = int(arr.shape[-1])
    if n == 0:
        return []
    max_len = int(max_sec * sample_rate)
    if max_len <= 0 or n <= max_len:
        return [AudioChunk(arr, 0.0, n / sample_rate)]

    overlap = max(0, int(overlap_sec * sample_rate))
    step = max_len - overlap
    if step <= 0:  # pathological overlap ≥ chunk — fall back to non-overlapping
        step = max_len

    chunks: list[AudioChunk] = []
    start = 0
    while start < n:
        end = min(start + max_len, n)
        chunks.append(AudioChunk(arr[start:end], start / sample_rate, end / sample_rate))
        if end >= n:
            break
        start += step
    return chunks


def float_to_wav_bytes(samples: "object", sample_rate: int = GEMMA_STT_SAMPLE_RATE) -> bytes:
    """Encode a 1-D float32 [-1,1] signal as a mono 16-bit PCM WAV container.

    Used to hand a chunk to llama-server over HTTP (which wants a decodable audio
    container). Pure stdlib (`wave`) — testable without any audio backend.
    """
    import wave

    import numpy as np

    arr = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


def detect_language(text: str) -> "tuple[str, float]":
    """Cheap script-based language guess (ru/en) with a confidence in [0,1].

    Gemma's ASR output is plain text with no language tag, so we infer it from the
    Cyrillic-vs-Latin letter ratio. Good enough to populate TranscriptionResult;
    not a substitute for real language ID on mixed-language input.
    """
    if not text:
        return "en", 0.0
    cyr = lat = 0
    for ch in text.lower():
        if "а" <= ch <= "я" or ch == "ё":
            cyr += 1
        elif "a" <= ch <= "z":
            lat += 1
    total = cyr + lat
    if total == 0:
        return "en", 0.0
    if cyr >= lat:
        return "ru", cyr / total
    return "en", lat / total


def _decode_audio(audio_bytes: bytes) -> "tuple[object, int]":
    """Decode arbitrary audio bytes (wav/webm-opus/ogg/mp3/…) to float32 PCM.

    Uses PyAV (ffmpeg) — the same decoder faster-whisper relies on — so it handles
    whatever MediaRecorder produces in the browser. Returns (samples, src_sr) where
    `samples` is float32 shaped (channels, n) at the file's NATIVE rate; mono /
    resample / normalize are applied afterwards by the pure functions above (so those
    steps stay testable in isolation).
    """
    import av  # type: ignore[import-untyped]
    import numpy as np

    with av.open(io.BytesIO(audio_bytes)) as container:
        if not container.streams.audio:
            raise ValueError("no audio stream found in input bytes")
        astream = container.streams.audio[0]
        src_sr = int(astream.rate or GEMMA_STT_SAMPLE_RATE)
        # Convert sample format to planar float32 but KEEP native rate + channel layout.
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        blocks: list = []

        def _emit(frame) -> None:
            out = resampler.resample(frame)
            # PyAV returns a list (>=9) or a single frame/None (older) — normalize.
            if out is None:
                return
            for rf in (out if isinstance(out, list) else [out]):
                blocks.append(rf.to_ndarray())  # (channels, n) planar float32

        for frame in container.decode(astream):
            _emit(frame)
        # Flush the resampler (newer PyAV); ignore if unsupported.
        try:
            _emit(None)
        except (ValueError, TypeError):
            pass

    if not blocks:
        return np.zeros((1, 0), dtype=np.float32), src_sr
    return np.concatenate(blocks, axis=1).astype(np.float32), src_sr


class GemmaSTT:
    """
    Speech-to-Text via Gemma-4 E4B native audio (multimodal ASR).

    Our fine-tuned E4B is multimodal (text + vision + audio); vision/audio layers are
    FROZEN during QLoRA (`finetune_vision_layers=False`) and therefore preserved, so
    the base ASR capability survives the fine-tune. Routing STT through E4B removes the
    separate faster-whisper weights + process (−1 stack component) and is the
    Google-native / GEAR-aligned option. See `.system/gemma4_finetune_reference.md` §3b.

    LIVE (2026-07-06): GemmaSTT spawns its OWN `llama-server --mmproj` child (base
    gemma-4-E4B GGUF + BF16 audio mmproj, both from ggml-org) on CPU by default, so it
    never competes with the LLM router for the GPU. Audio is preprocessed to Gemma's
    contract (16 kHz mono, ≤30 s chunks) then transcribed via the OpenAI-compatible
    `/v1/chat/completions` `input_audio` part. ~1.5 s per short clip.

    Uses the BASE gemma-4 model, NOT the synapse fine-tune: the persona fine-tune
    ignores "transcribe verbatim" and answers in character (verified). The ASR
    instruction is sent BEFORE the audio (text-first) — audio-first makes the model
    reply to the speech instead of transcribing it. See config.GEMMA_STT_*.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._logfile = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    # ── llama-server (mmproj) lifecycle ──────────────────────────────────────
    def _ensure_server(self) -> None:
        """Lazily spawn the `llama-server --mmproj` audio child + wait for /health."""
        if self._ready and self._proc and self._proc.poll() is None:
            return
        with self._load_lock:
            if self._ready and self._proc and self._proc.poll() is None:
                return
            if self._load_error:
                raise RuntimeError(f"STT server failed to load previously: {self._load_error}")
            for p, label in ((GEMMA_STT_MODEL_PATH, "STT model"),
                             (GEMMA_STT_MMPROJ_PATH, "audio mmproj"),
                             (str(LLAMA_SERVER_EXE), "llama-server")):
                if not os.path.exists(p):
                    self._load_error = f"{label} not found: {p}"
                    raise FileNotFoundError(self._load_error)
            self._spawn()

    def _spawn(self) -> None:
        args = [
            str(LLAMA_SERVER_EXE),
            "--model", str(GEMMA_STT_MODEL_PATH),
            "--mmproj", str(GEMMA_STT_MMPROJ_PATH),
            "--n-gpu-layers", str(GEMMA_STT_GPU_LAYERS),
            "--ctx-size", str(GEMMA_STT_CTX_SIZE),
            "--jinja",                       # gemma-4 chat template (required for audio)
            "--host", GEMMA_STT_HOST,
            "--port", str(GEMMA_STT_PORT),
        ]
        logger.info("Spawning GemmaSTT llama-server: %s + %s (gpu_layers=%d)",
                    Path(GEMMA_STT_MODEL_PATH).name, Path(GEMMA_STT_MMPROJ_PATH).name, GEMMA_STT_GPU_LAYERS)
        log_path = Path(__file__).resolve().parent / "llama-server.stt.log"
        try:
            self._logfile = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError:
            self._logfile = subprocess.DEVNULL  # type: ignore[assignment]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(
                args, cwd=str(LLAMA_BIN_DIR),
                stdout=self._logfile, stderr=subprocess.STDOUT,
                env=child_env(), creationflags=creationflags,
            )
        except Exception as e:
            self._load_error = str(e)
            self._close_log()
            logger.error("Failed to spawn STT server: %s", _safe_err(e))
            raise

        import requests  # type: ignore[import-untyped]
        deadline = time.monotonic() + GEMMA_STT_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._close_log(); self._proc = None
                self._load_error = "STT llama-server exited during startup (see llama-server.stt.log)"
                raise RuntimeError(self._load_error)
            try:
                if requests.get(f"{GEMMA_STT_SERVER_URL}/health", timeout=2).status_code == 200:
                    self._ready = True
                    logger.info("GemmaSTT server healthy on %s", GEMMA_STT_SERVER_URL)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.unload()
        self._load_error = f"STT server not healthy within {GEMMA_STT_STARTUP_TIMEOUT:.0f}s"
        raise RuntimeError(self._load_error)

    def _close_log(self) -> None:
        if self._logfile not in (None, subprocess.DEVNULL):
            try:
                self._logfile.close()  # type: ignore[union-attr]
            except Exception:
                pass
        self._logfile = None

    # ── preprocessing (pure DSP, no model) ───────────────────────────────────
    def preprocess(self, audio_bytes: bytes) -> "tuple[list[AudioChunk], float]":
        """Decode + condition audio into Gemma-ready ≤30 s chunks.

        Returns (chunks, total_duration_sec). Runs fully offline — no model/GPU.
        """
        raw, src_sr = _decode_audio(audio_bytes)          # (channels, n) @ native sr
        mono = to_mono(raw)                               # (n,)
        mono16 = resample_linear(mono, src_sr, GEMMA_STT_SAMPLE_RATE)
        mono16 = normalize_peak(mono16)
        duration = len(mono16) / GEMMA_STT_SAMPLE_RATE
        # DoS guard: a tiny compressed file can decode to hours of audio → thousands of
        # serial STT chunks that monopolize the shared CPU STT server. Reject early.
        if duration > GEMMA_STT_MAX_DURATION_SEC:
            raise ValueError(
                f"Audio too long: {duration:.0f}s exceeds the {GEMMA_STT_MAX_DURATION_SEC}s limit"
            )
        chunks = chunk_audio(mono16, GEMMA_STT_SAMPLE_RATE,
                             GEMMA_STT_MAX_SEGMENT_SEC, GEMMA_STT_CHUNK_OVERLAP_SEC)
        return chunks, duration

    # ── model call ───────────────────────────────────────────────────────────
    def _transcribe_chunk(self, chunk: "AudioChunk") -> str:
        """Send one ≤30 s chunk to the mmproj llama-server and return its transcript.

        OpenAI-compatible POST /v1/chat/completions with the ASR instruction FIRST
        then an `input_audio` part (base64 WAV). Text-first is deliberate — see class doc.
        """
        import requests  # type: ignore[import-untyped]

        self._ensure_server()
        wav_bytes = float_to_wav_bytes(chunk.samples, GEMMA_STT_SAMPLE_RATE)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        payload = {
            "model": GEMMA_STT_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GEMMA_STT_PROMPT},
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    ],
                }
            ],
            "temperature": 0.0,   # deterministic transcription
            "stream": False,
        }

        resp = requests.post(
            f"{GEMMA_STT_SERVER_URL.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=GEMMA_STT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected llama-server response shape: {data!r}") from exc

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """
        Transcribe audio via Gemma-4 native audio input (self-hosted, llama.cpp mtmd).

        Pipeline: preprocess → per-chunk transcription → join → infer language.
        Same TranscriptionResult shape the rest of the server expects.
        """
        chunks, duration = self.preprocess(audio_bytes)
        parts = [self._transcribe_chunk(ch) for ch in chunks]
        text = " ".join(p for p in parts if p).strip()
        language, prob = detect_language(text)
        return TranscriptionResult(
            text=text,
            language=language,
            language_prob=prob,
            duration=duration,
        )

    @property
    def is_loaded(self) -> bool:
        """True if the mmproj llama-server child is alive + healthy."""
        proc = self._proc
        return bool(self._ready and proc is not None and proc.poll() is None)

    def unload(self) -> None:
        """Terminate the audio llama-server child and release it."""
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._close_log()
        logger.info("GemmaSTT server stopped")


# ── Factory ──────────────────────────────────────────────────────────────────
# One backend now (Whisper removed). Kept as a plain annotation so importing
# stt.py never forces a heavy model import.
STTEngine = "GemmaSTT"


# ── Google Cloud Speech-to-Text ──────────────────────────────────────────────


def _probe_audio_meta(audio_bytes: bytes) -> "tuple[float, int]":
    """Duration in seconds and sample rate in Hz, from the container header only.

    Both come from one open. The budget is charged in SECONDS, so the length has
    to be known BEFORE the request goes out — decoding the whole clip just to
    measure it would cost more than the transcription. The RATE has to be known
    because Google requires `sampleRateHertz` for the Opus encodings and refuses
    the request without it:

        Invalid recognition 'config': Opus sample rate (0) not in supported
        rates 8000, 12000, 16000, 24000, 48000

    That is a real 400 from the live API, returned by the first genuine call this
    backend ever made. It would have hit every voice request in production, where
    the browser sends WebM/Opus — the class docstring's warning that the request
    shape was never verified turned out to be worth exactly what it said.

    Returns 0 for either value when the header does not carry it: the caller
    treats an unknown duration as "charge a conservative estimate" and an unknown
    rate as "fall back to the encoding's natural rate".
    """
    try:
        import av  # type: ignore[import-untyped]

        with av.open(io.BytesIO(audio_bytes)) as container:
            astream = container.streams.audio[0] if container.streams.audio else None
            rate = int(getattr(astream, "rate", 0) or 0) if astream is not None else 0
            if container.duration:
                return float(container.duration) / 1_000_000.0, rate
            if astream is not None and astream.duration and astream.time_base:
                return float(astream.duration * astream.time_base), rate
            return 0.0, rate
    except Exception:  # noqa: BLE001 — a probe must never be the reason a request fails
        pass
    return 0.0, 0


def _sniff_opus_encoding(audio_bytes: bytes) -> str:
    """Pick Google's encoding enum from the container's magic bytes.

    The browser prefers audio/webm;codecs=opus and falls back to audio/ogg;
    codecs=opus (src/lib/synapse-audio.ts), and Google accepts both natively — so
    nothing is transcoded, which matters because the deployed server has no
    ffmpeg binary. Guessing wrong earns an opaque 400, so read the header rather
    than trusting the MIME type the browser claimed.
    """
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):   # EBML → WebM
        return "WEBM_OPUS"
    if audio_bytes.startswith(b"OggS"):
        return "OGG_OPUS"
    if audio_bytes.startswith(b"RIFF"):
        return "LINEAR16"
    # MP3 was missing, so it fell through to the WEBM_OPUS default below — which
    # is how a round-trip test holding Chirp's own MP3 earned an Opus error.
    # ID3v2 tag, or a bare MPEG frame sync (eleven set bits).
    if audio_bytes.startswith(b"ID3") or (
        len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0
    ):
        return "MP3"
    return "WEBM_OPUS"


class _SecondsBudget:
    """A monthly ceiling on billed audio seconds, persisted across restarts.

    Deliberately separate from tts.py's TTSBudget rather than generalised out of
    it: that class meters CHARACTERS and is live billing code that works. If a
    third quota ever appears, merge them then — not while wiring up the first
    cloud STT call.

    The lock covers read-modify-write so two concurrent voice requests cannot both
    see the same remaining balance and both spend it.
    """

    def __init__(self, path: str, limit_seconds: float) -> None:
        self._path = path
        self._limit = limit_seconds
        self._lock = threading.Lock()

    def _period(self) -> str:
        return time.strftime("%Y-%m")

    def _read(self) -> "tuple[str, float]":
        try:
            import json

            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            return str(data.get("period", "")), float(data.get("seconds", 0.0))
        except Exception:  # noqa: BLE001 — a missing/corrupt file means "nothing spent yet"
            return "", 0.0

    def charge(self, seconds: float) -> None:
        """Reserve `seconds` against this month, or raise if that exceeds the cap."""
        with self._lock:
            period, spent = self._read()
            if period != self._period():
                spent = 0.0
            if spent + seconds > self._limit:
                raise RuntimeError(
                    f"monthly speech-to-text budget exhausted "
                    f"({spent:.0f}s of {self._limit:.0f}s used this month)"
                )
            try:
                import json

                tmp = self._path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"period": self._period(), "seconds": spent + seconds}, fh)
                os.replace(tmp, self._path)
            except OSError as e:
                # Losing the ledger must not break transcription, but it must be
                # loud: an unpersisted budget silently resets on every restart.
                logger.warning("STT budget persist failed: %s", _safe_err(e))


class GoogleSTT:
    """Speech-to-text via Google Cloud, over REST.

    Why REST and not google-cloud-speech: that package is not installed on the
    deployed server, and adding one is a change only the owner can make. httpx and
    google-auth ARE installed (google-auth ships with google-cloud-texttospeech,
    which Chirp 3 already uses), and between them they cover minting a token and
    posting a request. Zero new dependencies.

    Shape matches GemmaSTT exactly so `get_stt()` can return either: a SYNCHRONOUS
    `transcribe(bytes)`, an `is_loaded` property that never touches the network,
    and an `unload()` that tolerates being called on an instance that never ran.
    main.py calls transcribe through `asyncio.to_thread`, and up to
    SYNAPSE_MAX_CONCURRENT_INFERENCE of those run at once, so the lazy client is
    built under a lock.

    NOT INDEPENDENTLY VERIFIED: the exact v1 request/response field names below
    come from Google's Speech-to-Text documentation, but the workflow's skeptic
    agents for this area all died on a session limit before they could check them
    against the live API. The first real call is therefore the actual test — treat
    a 400 here as "the shape is wrong", not "the audio is bad".
    """

    _SCOPE = "https://www.googleapis.com/auth/cloud-platform"

    def __init__(self) -> None:
        self._creds: object | None = None
        self._client: object | None = None
        self._lock = threading.Lock()
        self._budget = _SecondsBudget(GOOGLE_STT_BUDGET_FILE, GOOGLE_STT_BUDGET_SECONDS)

    @staticmethod
    def credentials_available() -> bool:
        """True when a service-account file is configured AND present on disk."""
        path = GOOGLE_STT_CREDENTIALS
        return bool(path) and Path(path).is_file()

    @property
    def is_loaded(self) -> bool:
        """Cheap and network-free — /api/health reads this on an unauthenticated path."""
        return self._client is not None

    def unload(self) -> None:
        """Drop the HTTP client and credentials. Safe on a never-loaded instance."""
        with self._lock:
            client = self._client
            self._client = None
            self._creds = None
        if client is not None:
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            logger.info("Google STT client released")

    def _ensure_client(self):
        """Build credentials + client once, under a lock (double-checked)."""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            if not self.credentials_available():
                raise RuntimeError(
                    "Google STT requires a service-account JSON; set "
                    "SYNAPSE_GOOGLE_STT_CREDENTIALS or SYNAPSE_GOOGLE_TTS_CREDENTIALS"
                )
            import httpx
            from google.oauth2 import service_account  # type: ignore[import-untyped]

            # google-auth refreshes the token itself when it expires; the code
            # only has to ask for a fresh one before each request (below).
            self._creds = service_account.Credentials.from_service_account_file(
                GOOGLE_STT_CREDENTIALS, scopes=[self._SCOPE]
            )
            self._client = httpx.Client(timeout=GOOGLE_STT_TIMEOUT)
            logger.info(
                "Google STT client initialized (model=%s, lang=%s, alts=%s)",
                GOOGLE_STT_MODEL,
                GOOGLE_STT_LANGUAGE,
                ",".join(GOOGLE_STT_ALT_LANGUAGES) or "-",
            )
            return self._client

    def _access_token(self) -> str:
        from google.auth.transport.requests import Request  # type: ignore[import-untyped]

        creds = self._creds
        if creds is None:
            raise RuntimeError("Google STT credentials not initialized")
        if not creds.valid:  # type: ignore[attr-defined]
            creds.refresh(Request())  # type: ignore[attr-defined]
        return str(creds.token)  # type: ignore[attr-defined]

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """Transcribe one clip. Raises on failure; returns empty text for silence.

        Returning an empty transcript is NOT an error path — main.py turns that
        into an honest "No speech detected" rather than a failure, which is what
        someone who tapped the mic and said nothing should see.
        """
        client = self._ensure_client()

        if len(audio_bytes) > GOOGLE_STT_MAX_BYTES:
            raise ValueError(
                f"audio is {len(audio_bytes) / 1048576:.1f} MB; the synchronous "
                f"recognize endpoint accepts about {GOOGLE_STT_MAX_BYTES / 1048576:.0f} MB"
            )

        duration, sample_rate = _probe_audio_meta(audio_bytes)
        if duration > GOOGLE_STT_MAX_SECONDS:
            raise ValueError(
                f"clip is {duration:.0f}s; the synchronous recognize endpoint tops out "
                f"near {GOOGLE_STT_MAX_SECONDS:.0f}s — record a shorter message"
            )
        # An unmeasurable clip still costs money, so charge a conservative
        # estimate rather than nothing.
        self._budget.charge(duration if duration > 0 else 10.0)

        encoding = _sniff_opus_encoding(audio_bytes)
        # Google REQUIRES a rate for the Opus encodings and validates it against
        # the stream for the rest, so send what the container actually reports.
        # 48000 is a fallback for the Opus cases only, and only because Opus is
        # always 48 kHz internally and the browser's MediaRecorder emits exactly
        # that — a guess that is right for the one case where a header might not
        # parse. Anything else sends no rate rather than a fabricated one.
        if not sample_rate and encoding in ("WEBM_OPUS", "OGG_OPUS"):
            sample_rate = 48000

        body = {
            "config": {
                "encoding": encoding,
                "languageCode": GOOGLE_STT_LANGUAGE,
                "enableAutomaticPunctuation": True,
                "model": GOOGLE_STT_MODEL,
            },
            "audio": {"content": base64.b64encode(audio_bytes).decode("ascii")},
        }
        if sample_rate:
            body["config"]["sampleRateHertz"] = sample_rate
        if GOOGLE_STT_ALT_LANGUAGES:
            body["config"]["alternativeLanguageCodes"] = GOOGLE_STT_ALT_LANGUAGES

        res = client.post(  # type: ignore[attr-defined]
            GOOGLE_STT_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {self._access_token()}"},
        )
        if res.status_code != 200:
            # The body can echo request content; _safe_err keeps credential paths
            # and anything key-shaped out of the log.
            raise RuntimeError(
                f"Google STT returned HTTP {res.status_code}: {_safe_err(res.text[:300])}"
            )

        data = res.json()
        results = data.get("results") or []
        parts: list[str] = []
        api_lang = ""
        for r in results:
            alts = r.get("alternatives") or []
            if alts and alts[0].get("transcript"):
                parts.append(str(alts[0]["transcript"]).strip())
            api_lang = api_lang or str(r.get("languageCode") or "")

        text = " ".join(p for p in parts if p).strip()

        # Language must be a BARE two-letter code: tts.py compares `lang == "ru"`
        # exactly, so "ru-RU" would silently select the English voice. Prefer what
        # Google reports, fall back to the existing text heuristic, and take the
        # probability from that heuristic either way — the API returns a
        # transcription confidence, not a language confidence, and presenting one
        # as the other would be a fabricated number.
        heuristic_lang, prob = detect_language(text) if text else ("en", 0.0)
        lang = (api_lang or heuristic_lang)[:2].lower()
        if lang not in ("ru", "en"):
            lang = "en"

        logger.info(
            "Google STT: %d result(s), %d chars, lang=%s (api=%s), %.1fs audio, %s @ %dHz",
            len(results), len(text), lang, api_lang or "-", duration, encoding, sample_rate,
        )
        return TranscriptionResult(
            text=text, language=lang, language_prob=float(prob), duration=float(duration)
        )


def get_stt() -> "GemmaSTT | GoogleSTT":
    """
    Construct the STT backend. Google-only: always GemmaSTT (Gemma-4 E4B native
    audio). config.STT_BACKEND is honored for forward-compat, but "gemma" is the
    only backend — any other value logs a warning and still returns GemmaSTT (it
    can never silently pick a non-Google engine). Construction is cheap; the model
    path lazy-loads on first use (and currently raises NotImplementedError until
    the audio-mmproj GGUF is exported).
    """
    backend = (STT_BACKEND or "gemma").strip().lower()

    if backend in ("google", "cloud", "chirp"):
        if GoogleSTT.credentials_available():
            logger.info("STT backend: google (Cloud Speech-to-Text, REST)")
            return GoogleSTT()
        # Do NOT silently fall through to Gemma here. On the deployed server the
        # Gemma model files do not exist, so "falling back" would trade a clear
        # credentials error for a FileNotFoundError at the first voice message —
        # which is exactly how this feature stayed broken unnoticed.
        logger.error(
            "SYNAPSE_STT_BACKEND=google but no service-account JSON was found "
            "(SYNAPSE_GOOGLE_STT_CREDENTIALS / SYNAPSE_GOOGLE_TTS_CREDENTIALS). "
            "Voice input will fail until one is configured."
        )
        return GoogleSTT()

    if backend != "gemma":
        logger.warning(
            "Unknown SYNAPSE_STT_BACKEND=%r — falling back to 'gemma'", STT_BACKEND
        )
    logger.info("STT backend: gemma (Gemma-4 E4B native audio)")
    return GemmaSTT()
