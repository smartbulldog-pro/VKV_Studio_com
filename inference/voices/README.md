# Voice References

Place a 6-second WAV file here as `synapse-voice.wav` for XTTS v2 voice cloning.

Requirements:
- Format: WAV (16-bit, mono or stereo)
- Duration: 5-10 seconds of clean speech
- Quality: Clear recording, minimal background noise
- Content: Natural speech (any language)

The XTTS v2 engine will clone the voice characteristics (timbre, pitch, style)
from this reference file and use them for all generated speech.

Without this file, XTTS falls back to a default voice.
