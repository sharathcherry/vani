"""stt.py — Speech-to-Text and audio utilities.

Handles the full audio ingestion pipeline:
  1. download_twilio_audio() — fetch audio from Twilio CDN
  2. convert_to_wav()        — transcode any format to 16 kHz mono PCM WAV via ffmpeg
  3. detect_lang_from_script() — Unicode script scan for Indian language detection
  4. azure_stt()             — Azure AI Speech SDK push-stream STT with 3-layer LID

Phase 3 migration (2026-06): Replaces AWS Transcribe + S3 round-trip with Azure
AI Speech SDK push-stream recognition. ffmpeg 16 kHz mono PCM normalization is
preserved. The Twilio webhook contract is unchanged.

Language detection uses three layers in priority order:
  Layer 1 — Azure auto-detect LID (up to 4 candidate languages, SDK limit)
  Layer 2 — Unicode script scan on transcript (definitive for Indian scripts)
  Layer 3 — Fallback en-IN
"""

import os
import re
import subprocess
import tempfile
from urllib.parse import urljoin, urlsplit

import httpx

import azure.cognitiveservices.speech as speechsdk

from config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_MAP,
    TWILIO_SID,
    TWILIO_TOKEN,
)

# Unicode character ranges for each supported Indian script.
# Te-IN is listed first because it is the highest-priority language for
# this deployment (Telangana/AP focus). Checked in order; first script
# whose character density exceeds 15% wins.
_SCRIPT_DETECT = [
    ("te-IN", "ఀ", "౿"),   # Telugu        ← #1 priority
    ("kn-IN", "ಀ", "೿"),   # Kannada
    ("hi-IN", "ऀ", "ॿ"),   # Devanagari (Hindi / Marathi)
    ("ta-IN", "஀", "௿"),   # Tamil
    ("ml-IN", "ഀ", "ൿ"),   # Malayalam
    ("gu-IN", "઀", "૿"),   # Gujarati
    ("pa-IN", "਀", "੿"),   # Gurmukhi (Punjabi)
    ("or-IN", "଀", "୿"),   # Odia
    ("bn-IN", "ঀ", "৿"),   # Bengali / Assamese
    ("as-IN", "ঀ", "৿"),
]


def download_twilio_audio(media_url: str) -> tuple[bytes, str]:
    """Download a voice note from Twilio CDN.

    Returns (audio_bytes, content_type).
    Uses Basic auth only for Twilio hosts and follows redirects manually.

    This avoids leaking credentials to non-Twilio redirect targets while still
    handling Twilio-internal redirect chains that may require auth.
    """
    if not TWILIO_SID or not TWILIO_TOKEN:
        raise RuntimeError(
            "Twilio credentials missing: set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
        )

    # Helpful diagnostic for subaccount credential mismatches.
    m = re.search(r"/Accounts/(AC[0-9a-fA-F]{32})/", media_url)
    media_account_sid = m.group(1) if m else ""
    if media_account_sid and media_account_sid != TWILIO_SID:
        print(
            f"[download] WARNING: media URL account SID ({media_account_sid}) "
            f"!= configured TWILIO_ACCOUNT_SID ({TWILIO_SID}). "
            "If this is a subaccount flow, use that subaccount SID/token."
        )

    def _is_twilio_host(host: str) -> bool:
        host = (host or "").lower()
        return host.endswith("twilio.com") or host.endswith("twilio.io")

    next_url = media_url
    max_hops = 6
    with httpx.Client(timeout=10.0, follow_redirects=False) as client:
        for _ in range(max_hops):
            host = urlsplit(next_url).netloc
            auth = (TWILIO_SID, TWILIO_TOKEN) if _is_twilio_host(host) else None
            resp = client.get(next_url, auth=auth)

            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location")
                if not loc:
                    raise RuntimeError("Twilio media redirect without Location header.")
                next_url = urljoin(next_url, loc)
                continue

            if resp.status_code == 401:
                raise RuntimeError(
                    "Twilio media fetch failed with 401 Unauthorized. "
                    "Check TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN (and subaccount creds if used)."
                )

            resp.raise_for_status()
            data = resp.content
            content_type = resp.headers.get("Content-Type", "audio/ogg; codecs=opus")
            print(f"[download] {len(data)} bytes, Content-Type: {content_type}")
            return data, content_type

    raise RuntimeError("Too many redirects while downloading Twilio media.")


def convert_to_wav(audio_bytes: bytes) -> bytes:
    """Transcode any audio format to 16 kHz mono PCM WAV.

    Uses the ffmpeg binary bundled as a Lambda layer at /opt/bin/ffmpeg
    (or the PATH on Azure Container Apps). Azure Speech SDK push streams
    accept 16 kHz mono PCM; WAV header is stripped by the SDK file reader.
    """
    ffmpeg = os.environ.get("FFMPEG_BINARY", "/opt/bin/ffmpeg")
    if not os.path.exists(ffmpeg):
        # Fall back to PATH lookup (Azure Container Apps / local dev)
        ffmpeg = "ffmpeg"

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False, dir="/tmp") as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", in_path, "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", "-f", "wav", out_path],
            capture_output=True,
            timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exit {result.returncode}: {result.stderr.decode()[:200]}"
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def detect_lang_from_script(text: str, fallback: str = "en-IN") -> str:
    """Identify language by Unicode script character density.

    Checks which script block accounts for ≥15% of non-whitespace
    characters. This is the ground-truth layer 2 fallback when Azure LID
    either returns empty or echoes the primary hint without confidence.
    """
    if not text:
        return fallback

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return fallback

    total = len(non_space)
    best_lang, best_count = fallback, 0
    seen: set[str] = set()

    for lang, lo, hi in _SCRIPT_DETECT:
        if lang in seen:
            continue
        seen.add(lang)
        count = sum(1 for c in non_space if lo <= c <= hi)
        if count > best_count and (count / total) >= 0.15:
            best_count, best_lang = count, lang

    return best_lang


def _write_wav_temp(wav_bytes: bytes) -> str:
    """Persist WAV bytes to /tmp and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as f:
        f.write(wav_bytes)
        return f.name


def azure_stt(
    audio_bytes: bytes,
    content_type: str,
    preferred_lang: str = "te-IN",
) -> tuple[str, str]:
    """Azure AI Speech SDK STT with automatic language detection.

    Pipeline:
      Twilio mp3/wav/ogg → ffmpeg → 16 kHz mono PCM WAV → Azure Speech SDK

    `preferred_lang` is the language detected in the user's previous turn,
    stored in the S3 session. It is rotated into slot 0 of the SDK's
    AutoDetectSourceLanguageConfig (max 4 candidates) so the correct script
    appears in the transcript on the first try.

    Returns (transcript, detected_lang_code).

    Detection priority:
      1. Azure auto-detect LID confidently identifies a candidate
      2. Unicode script scan on the returned transcript (Layer 2)
      3. Azure's result when transcript is Latin script
      4. Fallback to en-IN
    """
    if not AZURE_SPEECH_KEY:
        print("[STT] AZURE_SPEECH_KEY not set")
        return "", "en-IN"

    wav_path = ""
    try:
        # Azure Speech SDK accepts a WAV file directly via AudioConfig(filename=...).
        # The SDK reads the WAV RIFF header and decodes PCM internally.
        # For OGG/Opus we still ffmpeg-transcode to WAV because the SDK file reader
        # used here does not unpack Opus reliably across all platforms.
        if "wav" in content_type:
            wav_bytes = audio_bytes
        else:
            print(f"[STT] Transcoding {content_type} → 16 kHz mono PCM WAV ...")
            wav_bytes = convert_to_wav(audio_bytes)
            print(f"[STT] WAV size: {len(wav_bytes):,} bytes")

        wav_path = _write_wav_temp(wav_bytes)

        # Azure auto-detect LID hard limit: maximum 4 candidate languages.
        # Priority covers the most common Indian languages for this use case.
        # Tamil, Malayalam, Marathi etc. are caught by Unicode script scan (Layer 2).
        _LID_PRIORITY = ["te-IN", "kn-IN", "hi-IN", "en-IN"]
        if preferred_lang not in _LID_PRIORITY and preferred_lang in AZURE_VOICE_MAP:
            # Rotate session language into slot 0 so Azure has it as a candidate
            _LID_PRIORITY = [preferred_lang] + _LID_PRIORITY[:3]

        primary_lang = preferred_lang if preferred_lang in AZURE_VOICE_MAP else "te-IN"

        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION,
        )
        # Set primary language hint — used when LID is inconclusive.
        speech_config.speech_recognition_language = primary_lang

        auto_detect_cfg = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=_LID_PRIORITY
        )
        audio_input = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect_cfg,
            audio_config=audio_input,
        )

        # recognize_once_async returns within ~1-2 s for ≤30 s of audio.
        # For multi-turn streaming, use start_continuous_recognition with
        # a Recognized event handler (out of scope for this webhook path).
        result = recognizer.recognize_once_async().get()

        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            cancellation = (
                speechsdk.CancellationDetails.from_result(result).reason.name
                if result.reason == speechsdk.ResultReason.Canceled
                else result.reason.name
            )
            print(f"[STT] no speech recognized: {cancellation}")
            return "", "en-IN"

        transcript = result.text or ""
        auto_lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
        azure_lang = (auto_lang_result.language or "").strip()

        print(
            f"[STT] reason={result.reason.name} azure_lang={azure_lang!r} "
            f"text='{transcript[:70]}'"
        )

        if not transcript:
            return "", "en-IN"

        # For the 4 priority languages (te/kn/hi/en), trust Azure even when it
        # matches the primary hint — these are well-supported and reliable.
        # For all other languages, only trust Azure if it actively overrides the hint.
        _PRIORITY_LANGS = {"te-IN", "kn-IN", "hi-IN", "en-IN"}
        if azure_lang and azure_lang in AZURE_VOICE_MAP and (
            azure_lang != primary_lang or azure_lang in _PRIORITY_LANGS
        ):
            detected = azure_lang
            print(f"[STT] lang={detected} (Azure LID)")
        else:
            script_lang = detect_lang_from_script(transcript, fallback="en-IN")
            if script_lang != "en-IN":
                detected = script_lang
                print(f"[STT] lang={detected} (Unicode script; azure_lang={azure_lang!r})")
            elif azure_lang and azure_lang in AZURE_VOICE_MAP:
                detected = azure_lang
                print(f"[STT] lang={detected} (Azure; no Indian script in transcript)")
            else:
                detected = "en-IN"
                print("[STT] lang=en-IN (fallback; no reliable detection)")

        return transcript, detected

    except Exception as e:
        print(f"[STT] Azure SDK failed: {e}")
        return "", "en-IN"
    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
