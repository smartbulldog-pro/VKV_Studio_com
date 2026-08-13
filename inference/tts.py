"""
Synapse TTS — Text-to-Speech, consolidated around Google.
=========================================================
Two backends behind one interface + factory:

  - ChirpTTS      : Google Cloud Text-to-Speech "Chirp 3: HD" — PRIMARY.
                    Google-native (GEAR). $30 / 1M chars, free 1M chars/month.
  - EdgeTTSEngine : Microsoft Edge TTS — free, no API key. The automatic FALLBACK
                    when Google credentials are missing/invalid, and the dev default.

Switch with SYNAPSE_TTS_BACKEND ("chirp3" | "edge"), default "chirp3". If "chirp3"
is selected but no Google credentials are found, the factory logs a WARNING and
returns Edge instead — voice never hard-fails in dev without a key (same philosophy
as the frontend mock when the backend is down). ChirpTTS also falls back to Edge
per-request if a synth call raises, so a transient Google error can't drop a reply.

TTS is always a SEPARATE step from STT/LLM: an LLM emits TEXT, not audio, so the
voice-out stage is its own engine — it pairs with (future) Gemma-native STT on the
hear side but stays independent. See inference/TTS_BACKEND.md.

Removed (2026-07-05): Coqui XTTS v2 + Silero (local model cruft) and the old
Neural2 Google engine — folded into ChirpTTS.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from abc import ABC, abstractmethod

from config import (
    CHIRP_TTS_VOICE_EN,
    CHIRP_TTS_VOICE_RU,
    GOOGLE_TTS_CREDENTIALS,
    TTS_BACKEND,
    TTS_BUDGET_FILE,
    TTS_MAX_CHARS,
    TTS_MONTHLY_CHAR_CAP,
    TTS_OUTPUT_FORMAT,
    TTS_SAMPLE_RATE,
)
from tts_budget import TTSBudget

logger = logging.getLogger("synapse.tts")


# ── Secret hygiene + cost helpers ────────────────────────────────────────────

def _scrub_secret(value: object) -> str:
    """Redact the credentials PATH from any string before it is logged/returned.

    The key file CONTENT is never read into memory as a string, so it cannot leak;
    this is belt-and-suspenders for the path (which is otherwise safe to log).
    """
    s = str(value)
    cred = GOOGLE_TTS_CREDENTIALS
    if cred:
        s = s.replace(cred, "<tts-creds-path>")
    return s


def within_char_cap(text: str) -> bool:
    """True if `text` is within the per-request TTS character cap (SYNAPSE_TTS_MAX_CHARS)."""
    return len(text or "") <= TTS_MAX_CHARS


def check_credentials_location(repo_root: "str | None" = None) -> "str | None":
    """Return a WARNING string if the creds file sits INSIDE the repo tree, else None.

    Logs only the path (safe), never the content. Called at startup so a key
    accidentally kept in the repo is surfaced (move it out + rotate).
    """
    cred = GOOGLE_TTS_CREDENTIALS
    if not cred:
        return None
    from pathlib import Path
    try:
        cred_path = Path(cred).resolve()
        root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parent.parent
        cred_path.relative_to(root)  # raises ValueError if not inside
    except ValueError:
        return None
    except OSError:
        return None
    return (
        f"Google TTS credentials at '{cred_path}' are INSIDE the repo tree — "
        "move the key OUTSIDE the repo and ROTATE it (set SYNAPSE_GOOGLE_TTS_CREDENTIALS "
        "to an external path). See inference/TTS_BACKEND.md § Security & cost controls."
    )


# Lazily-created process-wide monthly budget (shared by all ChirpTTS instances).
_default_budget: "TTSBudget | None" = None
_default_budget_lock = threading.Lock()


def get_default_budget() -> TTSBudget:
    global _default_budget
    if _default_budget is None:
        with _default_budget_lock:
            if _default_budget is None:
                _default_budget = TTSBudget(TTS_BUDGET_FILE, TTS_MONTHLY_CHAR_CAP)
    return _default_budget


# ── Abstract Interface ───────────────────────────────────────────────────────

class TTSEngine(ABC):
    """Abstract TTS engine. All backends implement this interface."""

    @abstractmethod
    def synthesize(self, text: str, lang: str) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: The text to speak.
            lang: Language code ("ru" or "en").

        Returns:
            Encoded audio bytes in this engine's `output_format`.
        """

    @property
    def output_format(self) -> str:
        """Container/codec of synthesize() output: "mp3" | "wav" | "ogg".

        main.py maps this to the Content-Type and X-Synapse-Audio-Format header.
        Defaults to "wav"; engines override as needed.
        """
        return "wav"

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the engine is ready to synthesize."""

    @abstractmethod
    def unload(self) -> None:
        """Release resources."""


# ── Markdown → plain text for TTS (natural reading) ─────────────────────────

def _markdown_to_plain(text: str) -> str:
    """
    Convert markdown text to clean plain text for TTS.

    Neural TTS engines don't read markdown, so we normalize to speech-friendly text:
      **bold**  → the text (bold stripped, naturally read)
      *italic*  → the text (italic stripped)
      # Heading → "Heading." (period = natural pause)
      ```code``` → skipped entirely
      `inline`  → read without backticks
      - bullets → "Item." (period = pause between items)
      Emojis    → converted to prosody punctuation, then stripped
      Links     → stripped
    """
    import re

    # ── Emoji → prosody conversion ───────────────────────────────────────────
    # Neural TTS engines respond to punctuation cues:
    #   ! → excited/happy       ... → trailing off, suggestive
    #   ?! → surprised          хм, → thoughtful (Russian)
    #   ; → slight irony        hmm, → thoughtful (English)
    #
    # We replace emoji with punctuation that modifies the tone of the
    # preceding phrase, then strip any remaining emoji.

    # Map: emoji → (replacement punctuation, position)
    # 'suffix' = replace sentence-ending punctuation before emoji
    # 'prefix' = inject before the sentence containing the emoji
    _EMOJI_TONE: dict[str, str] = {
        # Joy / positive
        '😊': '!', '😄': '!', '😃': '!', '🙂': '!', '😁': '!',
        '🥰': '!', '❤': '!', '💛': '!', '💚': '!', '🎉': '!',
        '👍': '!', '✅': '!', '🔥': '!', '💪': '!', '⭐': '!',
        # Wink / playful / irony
        '😉': '...', '😏': '...', '🙃': '...', '😜': '...',
        '😎': '...', '🤫': '...',
        # Thinking / uncertainty
        '🤔': '...', '🧐': '...', '💭': '...',
        # Surprise
        '😮': '?!', '😲': '?!', '🤯': '?!', '😱': '?!', '🫢': '?!',
        # Sadness / concern
        '😢': '...', '😞': '...', '😔': '...', '🥺': '...',
        '😕': '...', '☹': '...',
        # Laughter
        '😂': '!', '🤣': '!', '😆': '!',
    }

    def _apply_emoji_tone(t: str) -> str:
        """Replace emoji with prosody-affecting punctuation."""
        for emoji, punct in _EMOJI_TONE.items():
            if emoji not in t:
                continue
            # Find each occurrence and modify surrounding punctuation
            parts = t.split(emoji)
            result_parts: list[str] = []
            for i, part in enumerate(parts):
                if i > 0 and part == '' and result_parts:
                    # Multiple emoji in a row — skip extras
                    continue
                if i > 0 and result_parts:
                    # Modify the end of the previous part
                    prev = result_parts[-1].rstrip()
                    if prev and prev[-1] in '.':
                        # Replace period with emotional punctuation
                        result_parts[-1] = prev[:-1] + punct + ' '
                    elif prev and prev[-1] not in '!?':
                        # Add emotional punctuation
                        result_parts[-1] = prev + punct + ' '
                result_parts.append(part)
            t = ''.join(result_parts)
        return t

    text = _apply_emoji_tone(text)

    # Strip any remaining emojis/symbols not in our map
    text = re.sub(r'[\U00010000-\U0010ffff☀-➿]', '', text)

    # ── Strip HTML tags ──────────────────────────────────────────────────────
    # The model sometimes emits raw HTML (<h2>, <p>, <br>, <li>…). Without this
    # the TTS literally SPEAKS "h 2 About VKVstudio h 2". Turn heading/block
    # boundaries into a sentence break (period = natural pause), then delete every
    # remaining tag so nothing tag-shaped is ever voiced.
    text = re.sub(r'(?i)</?(h[1-6]|p|div|ul|ol|li|tr|table|section|article|blockquote)\s*/?>', '. ', text)
    text = re.sub(r'(?i)<br\s*/?>', '. ', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove code blocks entirely
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove markdown links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bare URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove **bold** markers (keep text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # Remove *italic* markers (keep text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)

    # Remove ~~strikethrough~~ markers
    text = re.sub(r'~~(.+?)~~', r'\1', text)

    lines = text.split('\n')
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', stripped):
            continue

        # Headings: ## Title → "Title."
        if stripped.startswith('#'):
            heading_text = re.sub(r'^#+\s*', '', stripped).strip()
            if heading_text:
                # Add period if not ending with punctuation
                if heading_text[-1] not in '.!?:':
                    heading_text += '.'
                result.append(heading_text)
            continue

        # Bullet points: "- Item" → "Item."
        if stripped.startswith(('- ', '• ')) or (stripped.startswith('* ') and not stripped.startswith('**')):
            bullet_text = stripped.lstrip('-•* ').strip()
            if bullet_text and bullet_text[-1] not in '.!?:':
                bullet_text += '.'
            if bullet_text:
                result.append(bullet_text)
            continue

        # Numbered lists: "1. Item" → "Item."
        num_match = re.match(r'^\d+\.\s+(.+)', stripped)
        if num_match:
            item_text = num_match.group(1).strip()
            if item_text and item_text[-1] not in '.!?:':
                item_text += '.'
            result.append(item_text)
            continue

        result.append(stripped)

    clean = ' '.join(result)
    # Collapse multiple spaces/newlines
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Collapse runs of periods left by tag→". " substitution (". . ." → ". ")
    clean = re.sub(r'(?:\.\s*){2,}', '. ', clean).strip()
    return clean


def _merge_unspeakable(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Fold segments that contain nothing pronounceable into the one before them.

    `_split_mixed_text` emits the gaps BETWEEN adjacent English words as Russian
    segments, so an ordinary bilingual sentence —

        "…три инструмента: Tokenizer Profiler, Prompt Architect и Embedding Explorer."

    — yields a segment that is just ", ". Edge TTS rejects that outright with
    "No audio was received. Please verify that your parameters are correct.",
    which raised, aborted the whole mixed render, and dropped the caller back to
    a single voice. That warning was firing in production on essentially every
    bilingual answer, meaning the dual-voice feature never actually ran.

    The punctuation is appended to the previous segment rather than dropped, so
    the comma still lands as a pause in the voice that spoke the words before
    it, instead of becoming a hard cut between two clips.
    """
    merged: list[tuple[str, str]] = []
    for raw_text, seg_lang in segments:
        if not raw_text.strip():
            continue
        if any(ch.isalnum() for ch in raw_text):
            merged.append((raw_text, seg_lang))
        elif merged:
            prev_text, prev_lang = merged[-1]
            merged[-1] = (prev_text + raw_text, prev_lang)
        # A leading unspeakable fragment has no previous segment to attach to
        # and carries no words, so it is dropped.
    return merged


def _split_mixed_text(text: str) -> list[tuple[str, str]]:
    """
    Split text into segments by language: Russian ('ru') and English ('en').

    Returns list of (text_segment, lang_code) tuples.
    Consecutive same-language segments are merged.

    Example:
      "Привет, используем React и TypeScript для проекта"
      → [("Привет, используем ", "ru"), ("React", "en"), (" и ", "ru"),
         ("TypeScript", "en"), (" для проекта", "ru")]
    """
    import re

    # Check if text has both Cyrillic and Latin content
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text))
    has_latin = bool(re.search(r'[a-zA-Z]{2,}', text))

    if not has_cyrillic or not has_latin:
        # Pure single-language text
        lang = "ru" if has_cyrillic else "en"
        return [(text, lang)]

    # Split into tokens preserving whitespace and punctuation
    # Pattern captures: English words (2+ Latin chars) vs everything else
    segments: list[tuple[str, str]] = []
    # Match English word sequences (2+ Latin chars, possibly with digits/dots/hyphens)
    en_pattern = re.compile(r'[A-Za-z][A-Za-z0-9.\-]*(?:\s+[A-Za-z][A-Za-z0-9.\-]*)*')

    last_end = 0
    for match in en_pattern.finditer(text):
        start, end = match.start(), match.end()
        en_word = match.group(0)

        # Skip single-char matches
        if len(en_word.strip()) < 2:
            continue

        # Add Russian text before this match
        if start > last_end:
            ru_part = text[last_end:start]
            if ru_part:
                segments.append((ru_part, "ru"))

        segments.append((en_word, "en"))
        last_end = end

    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            segments.append((remaining, "ru"))

    # Merge consecutive segments of same language
    if not segments:
        return [(text, "ru")]

    merged: list[tuple[str, str]] = [segments[0]]
    for seg_text, seg_lang in segments[1:]:
        if seg_lang == merged[-1][1]:
            merged[-1] = (merged[-1][0] + seg_text, seg_lang)
        else:
            merged.append((seg_text, seg_lang))

    return merged


# ── Google Cloud TTS — "Chirp 3: HD" (PRIMARY) ───────────────────────────────

class ChirpTTS(TTSEngine):
    """
    Google Cloud Text-to-Speech — "Chirp 3: HD" voices (PRIMARY / GEAR).

    Pricing: $30 / 1M characters, free 1M chars/month (billed per character incl.
    spaces) — effectively free at our traffic.

    Auth: a service-account JSON. Point SYNAPSE_GOOGLE_TTS_CREDENTIALS at the file
    (it is exported to GOOGLE_APPLICATION_CREDENTIALS), or set
    GOOGLE_APPLICATION_CREDENTIALS yourself. Package: `google-cloud-texttospeech`.

    Robustness: if a synth call fails (bad creds surfaced on first call, network,
    quota) it falls back to Edge TTS for that request so voice never hard-fails.
    """

    # Our format token → Google AudioEncoding enum name.
    _ENCODING = {"mp3": "MP3", "wav": "LINEAR16", "ogg": "OGG_OPUS"}

    def __init__(self, budget: "TTSBudget | None" = None) -> None:
        self._client = None
        self._texttospeech = None
        self._load_lock = threading.Lock()
        self._fallback: EdgeTTSEngine | None = None
        fmt = (TTS_OUTPUT_FORMAT or "mp3").lower()
        self._format = fmt if fmt in self._ENCODING else "mp3"
        # Monthly char budget (cost cap). Shared process-wide singleton by default;
        # injectable for tests. Keeps the backend inside Google's free tier → $0.
        self._budget = budget if budget is not None else get_default_budget()

    @staticmethod
    def credentials_available() -> bool:
        """True if a usable service-account JSON path is configured and exists."""
        cred = GOOGLE_TTS_CREDENTIALS or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        return bool(cred) and os.path.exists(cred)

    @property
    def output_format(self) -> str:
        return self._format

    def _ensure_loaded(self) -> None:
        """Initialize the Google Cloud TTS client, thread-safe."""
        if self._client is not None:
            return
        with self._load_lock:
            if self._client is not None:
                return
            # Export the service-account path so the client picks it up.
            if GOOGLE_TTS_CREDENTIALS and os.path.exists(GOOGLE_TTS_CREDENTIALS):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_TTS_CREDENTIALS

            from google.cloud import texttospeech  # type: ignore[import-untyped]

            self._client = texttospeech.TextToSpeechClient()
            self._texttospeech = texttospeech
            logger.info("Google Chirp 3 HD TTS client initialized (format=%s)", self._format)

    def _edge_fallback(self) -> "EdgeTTSEngine":
        if self._fallback is None:
            self._fallback = EdgeTTSEngine()
        return self._fallback

    def synthesize(self, text: str, lang: str) -> bytes:
        """Synthesize via Chirp 3 HD; fall back to Edge on budget-cap or any failure."""
        clean = _markdown_to_plain(text)
        if not clean:
            return b""

        n_chars = len(clean)  # what Google would bill (per character, incl. spaces)

        # ── Cost cap: atomically reserve the chars against the monthly budget.
        #    If the reservation fails (would cross the cap), DON'T call Google —
        #    use free Edge TTS so we physically stay in the free tier ($0). Doing
        #    check+reserve under one lock (try_reserve) closes the TOCTOU where
        #    concurrent requests could all pass the check and overshoot the cap. ──
        if self._budget is not None and not self._budget.try_reserve(n_chars):
            logger.warning(
                "TTS monthly char budget reached (%d/%d used) — routing to free Edge TTS "
                "to stay in Google's free tier ($0)",
                self._budget.usage(), self._budget.cap,
            )
            return self._edge_fallback().synthesize(text, lang)

        try:
            audio = self._synthesize_google(clean, lang)
        except Exception as e:  # noqa: BLE001 — degrade, never drop the reply
            # Nothing was actually sent to Google → refund the reservation so a
            # transient failure doesn't permanently consume budget.
            if self._budget is not None:
                self._budget.refund(n_chars)
            # Log a SCRUBBED, generic message server-side; never expose creds/stack
            # to the caller (the endpoint returns a generic message too).
            logger.warning(
                "Chirp 3 HD synth failed (%s) — falling back to Edge TTS for this request",
                _scrub_secret(e),
            )
            # Edge emits MP3; consistent with the default output_format ("mp3").
            return self._edge_fallback().synthesize(text, lang)

        return audio

    def _synthesize_google(self, clean_text: str, lang: str) -> bytes:
        self._ensure_loaded()
        # Local references: a concurrent unload() (e.g. /api/session/close) can null
        # self._client / self._texttospeech mid-call — bind them here so an in-flight
        # synthesis finishes against the live client instead of crashing.
        client = self._client
        tts = self._texttospeech
        assert client is not None and tts is not None

        if lang == "ru":
            voice_name = CHIRP_TTS_VOICE_RU
            language_code = "ru-RU"
        else:
            voice_name = CHIRP_TTS_VOICE_EN
            language_code = "en-US"

        synthesis_input = tts.SynthesisInput(text=clean_text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )
        audio_config = tts.AudioConfig(
            audio_encoding=getattr(tts.AudioEncoding, self._ENCODING[self._format]),
            sample_rate_hertz=TTS_SAMPLE_RATE,
        )
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        return response.audio_content

    @property
    def is_loaded(self) -> bool:
        if self._client is not None:
            return True
        return self._fallback is not None and self._fallback.is_loaded

    def unload(self) -> None:
        if self._client is not None:
            del self._client
            self._client = None
            self._texttospeech = None
            logger.info("Chirp 3 HD TTS client released")
        if self._fallback is not None:
            self._fallback.unload()
            self._fallback = None


# ── Microsoft Edge TTS (free fallback / dev default, ONLINE) ─────────────────

class EdgeTTSEngine(TTSEngine):
    """
    Microsoft Edge TTS — free, high-quality neural voices.

    No API key required. Excellent Russian and English voices.
    REQUIRES INTERNET — async API wrapped for sync usage.

    For mixed Russian+English text, splits into segments and synthesizes
    each with the appropriate native voice for perfect pronunciation.
    Outputs MP3.
    """
    # Native Russian voice — excellent Russian, passable English
    VOICE_RU = "ru-RU-SvetlanaNeural"
    # Multilingual voice — excellent English, good Russian
    VOICE_EN = "en-US-AvaMultilingualNeural"

    def __init__(self) -> None:
        self._loaded = True  # No model loading needed

    @property
    def output_format(self) -> str:
        return "mp3"

    def synthesize(self, text: str, lang: str) -> bytes:
        """Synthesize speech using Edge TTS. Strips markdown to clean text.

        For Russian text with English words, uses dual-voice synthesis:
        Russian segments → SvetlanaNeural, English segments → AvaMultilingualNeural.
        Audio is concatenated seamlessly.
        """
        import asyncio

        voice = self.VOICE_RU if lang == "ru" else self.VOICE_EN

        async def _synthesize() -> bytes:
            clean_text = _markdown_to_plain(text)
            if not clean_text:
                return b""

            # For Russian: try dual-voice if text has mixed languages
            if lang == "ru":
                segments = _split_mixed_text(clean_text)
                if len(segments) > 1:
                    try:
                        return await self._synthesize_mixed(segments)
                    except Exception as e:
                        # Fallback to single voice if dual-voice fails (e.g. network)
                        logger.warning("Mixed TTS failed, falling back to single voice: %s", e)

            # Pure single-language text or fallback — simple synthesis
            return await self._synthesize_single(clean_text, voice)

        # Run async in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _synthesize())
                return future.result()
        else:
            return asyncio.run(_synthesize())

    @staticmethod
    async def _synthesize_single(text: str, voice: str) -> bytes:
        """Synthesize a single text segment with one voice."""
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()

    async def _synthesize_mixed(self, segments: list[tuple[str, str]]) -> bytes:
        """Synthesize mixed-language segments with appropriate voices and concatenate."""
        import edge_tts

        all_audio = io.BytesIO()
        for text_part, seg_lang in _merge_unspeakable(segments):
            text_part = text_part.strip()
            if not text_part:
                continue

            voice = self.VOICE_RU if seg_lang == "ru" else self.VOICE_EN
            communicate = edge_tts.Communicate(text_part, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    all_audio.write(chunk["data"])

        return all_audio.getvalue()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False
        logger.info("Edge TTS unloaded")


# ── Factory ──────────────────────────────────────────────────────────────────

def create_tts_engine(backend: str | None = None) -> TTSEngine:
    """
    Create a TTS engine based on the configured backend.

    Args:
        backend: "chirp3" (Google Chirp 3 HD, primary) or "edge" (free fallback).
                 Defaults to SYNAPSE_TTS_BACKEND.

    Returns:
        A TTSEngine instance. If "chirp3" is requested without Google credentials,
        logs a WARNING and returns EdgeTTSEngine — voice keeps working in dev.
    """
    backend = (backend or TTS_BACKEND or "chirp3").strip().lower()

    if backend in ("chirp3", "chirp", "google", "google_cloud"):
        if ChirpTTS.credentials_available():
            logger.info("TTS backend: Google Chirp 3 HD (primary)")
            return ChirpTTS()
        logger.warning(
            "TTS backend '%s' requested but no Google credentials found "
            "(set SYNAPSE_GOOGLE_TTS_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS to a "
            "service-account JSON) — falling back to free Edge TTS.", backend,
        )
        return EdgeTTSEngine()

    if backend == "edge":
        logger.info("TTS backend: Microsoft Edge TTS (free)")
        return EdgeTTSEngine()

    logger.warning("Unknown TTS backend %r — falling back to Edge TTS", backend)
    return EdgeTTSEngine()
