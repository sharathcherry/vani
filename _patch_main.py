content = '''\
import json
import base64
import urllib.parse
import urllib.request
import boto3
import httpx
import time
import os
import subprocess
import tempfile
import threading

# --- Configuration ---
S3_BUCKET_IN  = os.environ.get("S3_VOICE_INPUT_BUCKET",  "whatsapp-voice-messages")
S3_BUCKET_OUT = os.environ.get("S3_VOICE_OUTPUT_BUCKET", "whatsapp-voice-responses")
RAG_LAMBDA_NAME = "gov-schemes-voice-rag"
REGION = "eu-north-1"

TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

# -- Azure Speech ----------------------------------------------------------
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")

# Neural voice map - covers every major Indian language supported by Azure
AZURE_VOICE_MAP = {
    "en-IN": ("en-IN", "en-IN-NeerjaNeural"),
    "hi-IN": ("hi-IN", "hi-IN-SwaraNeural"),
    "te-IN": ("te-IN", "te-IN-ShrutiNeural"),
    "ta-IN": ("ta-IN", "ta-IN-PallaveNeural"),
    "kn-IN": ("kn-IN", "kn-IN-SapnaNeural"),
    "ml-IN": ("ml-IN", "ml-IN-SobhanaNeural"),
    "mr-IN": ("mr-IN", "mr-IN-AarohiNeural"),
    "bn-IN": ("bn-IN", "bn-IN-TanishaaNeural"),
    "gu-IN": ("gu-IN", "gu-IN-DhwaniNeural"),
    "pa-IN": ("pa-IN", "pa-IN-VaaniNeural"),
    "or-IN": ("or-IN", "or-IN-SubhasiniNeural"),
    "as-IN": ("as-IN", "as-IN-YashicaNeural"),
}

# --- AWS Clients ---
s3            = boto3.client("s3", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
_bedrock      = boto3.client("bedrock-runtime", region_name="eu-central-1")

_LANG_NAMES = {
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
    "as-IN": "Assamese",
    "en-IN": "English",
}


def translate_text(text: str, source_lang_code: str, target_lang_code: str) -> str:
    """Translate text between languages using Bedrock Nova Lite."""
    src = _LANG_NAMES.get(source_lang_code, "English")
    tgt = _LANG_NAMES.get(target_lang_code, "English")
    if src == tgt or not text.strip():
        return text
    prompt = (
        f"Translate the following {src} text to {tgt}. "
        f"Output ONLY the translation, nothing else.\\n\\n{text}"
    )
    try:
        resp = _bedrock.converse(
            modelId="eu.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 512},
        )
        translated = resp["output"]["message"]["content"][0]["text"].strip()
        print(f"[translate] {src} -> {tgt}: \'{translated[:80]}\'")
        return translated if translated else text
    except Exception as e:
        print(f"[translate] failed ({e}), returning original")
        return text


def parse_body(event: dict) -> dict:
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    try:
        return dict(urllib.parse.parse_qsl(body))
    except Exception:
        return {}


def twiml_reply_media(audio_url: str) -> dict:
    escaped = audio_url.replace("&", "&amp;")
    xml = f\'\'\'<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Message>
            <Media>{escaped}</Media>
        </Message>
    </Response>
    \'\'\'
    return {"statusCode": 200, "headers": {"Content-Type": "text/xml"}, "body": xml}


def twiml_reply_text(message: str) -> dict:
    safe = message.replace("&", "&amp;").replace("<", "").replace(">", "")
    xml = f\'\'\'<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Message>{safe}</Message>
    </Response>
    \'\'\'
    return {"statusCode": 200, "headers": {"Content-Type": "text/xml"}, "body": xml}


class NoAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newreq = super().redirect_request(req, fp, code, msg, headers, newurl)
        if newreq is not None and "Authorization" in newreq.headers:
            del newreq.headers["Authorization"]
        if newreq is not None and "authorization" in newreq.headers:
            del newreq.headers["authorization"]
        return newreq


def _convert_to_wav(audio_bytes: bytes) -> bytes:
    """Convert any audio format to 16 kHz mono WAV via the ffmpeg Lambda layer."""
    ffmpeg = "/opt/bin/ffmpeg"
    if not os.path.exists(ffmpeg):
        raise RuntimeError("ffmpeg layer not attached - /opt/bin/ffmpeg not found")
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False, dir="/tmp") as f:
        f.write(audio_bytes)
        in_path = f.name
    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", in_path, "-ar", "16000", "-ac", "1",
             "-acodec", "pcm_s16le", "-f", "wav", out_path],
            capture_output=True, timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg exit {result.returncode}: {result.stderr.decode()[:200]}")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def _download_twilio_audio(media_url: str) -> tuple[bytes, str]:
    """Download audio from Twilio, stripping auth on redirect. Returns (bytes, content_type)."""
    auth_b64 = base64.b64encode(f"{TWILIO_SID}:{TWILIO_TOKEN}".encode()).decode()
    req = urllib.request.Request(media_url, headers={"Authorization": f"Basic {auth_b64}"})
    opener = urllib.request.build_opener(NoAuthRedirectHandler)
    with opener.open(req, timeout=10) as r:
        data = r.read()
        content_type = r.headers.get("Content-Type", "audio/ogg; codecs=opus")
    print(f"[download] {len(data)} bytes, Content-Type: {content_type}")
    return data, content_type


def _azure_stt(audio_bytes: bytes, content_type: str,
               preferred_lang: str = "en-IN") -> tuple[str, str]:
    """Azure Speech STT on already-downloaded bytes.

    preferred_lang (stored from user\'s previous turn) is used as the PRIMARY
    recognition language so Azure produces native script (e.g. Telugu characters,
    Punjabi Gurmukhi) instead of romanized transliteration when LID confidence
    is low.  Azure LID supports up to 10 candidate languages.
    Returns (transcript, detected_lang_code).
    """
    try:
        if "ogg" in content_type or "opus" in content_type or "wav" in content_type:
            audio_to_send    = audio_bytes
            stt_content_type = "audio/ogg; codecs=opus"
        else:
            print(f"[STT] Converting {content_type} -> WAV (ffmpeg)...")
            audio_to_send    = _convert_to_wav(audio_bytes)
            stt_content_type = "audio/wav; codecs=audio/pcm; samplerate=16000"
            print(f"[STT] WAV size: {len(audio_to_send):,} bytes")

        primary   = preferred_lang if preferred_lang in AZURE_VOICE_MAP else "en-IN"
        lid_langs = ",".join(list(AZURE_VOICE_MAP.keys())[:10])  # Azure cap = 10
        stt_url = (
            f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
            f"?language={primary}&lid={lid_langs}&format=simple"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": stt_content_type,
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(stt_url, headers=headers, content=audio_to_send)
        result     = resp.json()
        status     = result.get("RecognitionStatus", "Unknown")
        transcript = result.get("DisplayText", "")
        print(f"[STT] status={status} text=\'{transcript[:80]}\'")
        if status not in ("Success", "EndOfDictation"):
            print(f"[STT] full response: {result}")
        pl         = result.get("PrimaryLanguage", {})
        detected   = pl.get("Language", primary) if isinstance(pl, dict) else primary
        confidence = pl.get("Confidence", "?")   if isinstance(pl, dict) else "?"
        print(f"[STT] lang={detected} confidence={confidence}")
        return transcript, detected
    except Exception as e:
        print(f"[STT] Azure failed: {e}")
        return "", preferred_lang


def send_twilio_whatsapp(to_number: str, from_number: str, text: str = None, media_url: str = None):
    auth_b64 = base64.b64encode(f"{TWILIO_SID}:{TWILIO_TOKEN}".encode()).decode()
    data = {"From": from_number, "To": to_number}
    if media_url:
        data["MediaUrl"] = media_url
    if text:
        data["Body"] = text
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            headers={"Authorization": f"Basic {auth_b64}"},
            data=data,
        )
    if resp.status_code >= 400:
        print(f"[Twilio API] send failed {resp.status_code}: {resp.text}")
    else:
        print(f"[Twilio API] sent to {to_number}")


def synthesize_speech(text: str, message_sid: str, lang_code: str = "en-IN") -> str:
    """Azure Neural TTS -> S3 -> presigned URL for Twilio."""
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY environment variable is not set.")
    xml_lang, voice_name = AZURE_VOICE_MAP.get(lang_code, ("en-IN", "en-IN-NeerjaNeural"))
    print(f"[TTS] Azure Speech: lang={lang_code}, voice={voice_name}")
    safe_text = text[:2000].replace("&", "and").replace("<", "").replace(">", "")
    ssml = f\'\'\'<speak version=\'1.0\' xml:lang=\'{xml_lang}\'>
    <voice name=\'{voice_name}\'>{safe_text}</voice>
</speak>\'\'\'
    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
        "User-Agent": "GovSchemes-VoiceBot/1.0",
    }
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, headers=headers, content=ssml.encode("utf-8"))
    resp.raise_for_status()
    audio_bytes = resp.content
    print(f"[TTS] Azure returned {len(audio_bytes):,} bytes of MP3")
    s3_key = f"wa-out/{message_sid}.mp3"
    s3.put_object(Bucket=S3_BUCKET_OUT, Key=s3_key, Body=audio_bytes, ContentType="audio/mpeg")
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET_OUT, "Key": s3_key},
        ExpiresIn=3600,
    )
    return url


def get_rag_answer(english_query: str) -> str:
    response = lambda_client.invoke(
        FunctionName=RAG_LAMBDA_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps({
            "rawPath": "/debug/query",
            "body": json.dumps({"query": english_query}),
            "headers": {"content-type": "application/json"},
        }),
    )
    result = json.loads(response["Payload"].read())
    body   = json.loads(result.get("body", "{}"))
    return body.get("answer", "No information found.")


SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours


def get_session_context(phone_number: str) -> tuple[str, str]:
    """Returns (context_text, preferred_lang_code).

    preferred_lang_code is the language detected last turn.  It becomes the
    primary STT language next turn so Azure produces native script output.
    Session file format:  LANG:<code>\\nQ: ...\\nA: ...
    """
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_IN, Key=f"wa-sessions/{phone_number}.txt")
        age  = time.time() - resp["LastModified"].timestamp()
        if age > SESSION_TTL_SECONDS:
            print(f"[session] Expired ({age/3600:.1f}h old) - starting fresh.")
            s3.delete_object(Bucket=S3_BUCKET_IN, Key=f"wa-sessions/{phone_number}.txt")
            return "", "en-IN"
        raw = resp["Body"].read().decode("utf-8")
        if raw.startswith("LANG:"):
            first_line, _, rest = raw.partition("\\n")
            lang_code = first_line[5:].strip() or "en-IN"
            return rest, lang_code
        return raw, "en-IN"
    except Exception:
        return "", "en-IN"


def save_session_context(phone_number: str, english_query: str, answer: str, lang_code: str = "en-IN"):
    """Saves Q+A summary and detected language for next-turn STT."""
    context = f"LANG:{lang_code}\\nQ: {english_query[:200]}\\nA: {answer[:300]}"
    s3.put_object(
        Bucket=S3_BUCKET_IN,
        Key=f"wa-sessions/{phone_number}.txt",
        Body=context.encode("utf-8"),
    )


def lambda_handler(event, context):
    """Main Twilio WhatsApp Entry Point"""
    if event.get("_async_process"):
        _handle_voice_async(event)
        return {"statusCode": 200, "body": "ok"}

    params      = parse_body(event)
    message_sid = params.get("MessageSid", f"msg-{int(time.time())}")
    raw_phone   = params.get("From", "unknown")
    phone_id    = raw_phone.replace("+", "").replace(":", " ")
    media_url   = params.get("MediaUrl0")
    user_text   = params.get("Body", "").strip()

    if media_url:
        print(f"Processing Voice Note from: {media_url}")
        fn_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "vani-jan-webhook")
        lambda_client.invoke(
            FunctionName=fn_name,
            InvocationType="Event",
            Payload=json.dumps({
                "_async_process": True,
                "media_url":   media_url,
                "message_sid": message_sid,
                "from_number": params.get("From", ""),
                "to_number":   params.get("To", ""),
                "phone_id":    phone_id,
            }),
        )
        return twiml_reply_text("Got your voice note! Searching for schemes, reply in a moment...")

    user_query = user_text
    if not user_query:
        return twiml_reply_text("Please ask a question about government schemes via text or voice note.")

    print(f"User Query: {user_query}")
    last_context, _ = get_session_context(phone_id)
    rag_query = f"{last_context}\\nNew question: {user_query}" if last_context else user_query
    answer_text = get_rag_answer(rag_query)
    print(f"AI Answer: {answer_text}")
    save_session_context(phone_id, user_query, answer_text)
    return twiml_reply_text(answer_text)


def _handle_voice_async(event: dict):
    """Voice pipeline with threading for I/O parallelism.

    Parallelism:
    A) Session S3 read runs concurrently with Twilio audio download  (~1-2s saved)
    B) Session S3 write runs concurrently with Bedrock answer translation  (~1s saved)
    """
    media_url   = event["media_url"]
    message_sid = event["message_sid"]
    from_number = event["from_number"]
    to_number   = event["to_number"]
    phone_id    = event["phone_id"]

    # -- PARALLEL A: session read + audio download -------------------------
    session_result = ["", "en-IN"]   # [context_text, preferred_lang]

    def _load_session():
        ctx, lc = get_session_context(phone_id)
        session_result[0], session_result[1] = ctx, lc

    session_thread = threading.Thread(target=_load_session, daemon=True)
    session_thread.start()

    audio_bytes, content_type = _download_twilio_audio(media_url)
    try:
        s3.put_object(Bucket=S3_BUCKET_IN, Key=f"wa-in/{message_sid}.ogg", Body=audio_bytes)
    except Exception:
        pass

    session_thread.join(timeout=5.0)
    last_context   = session_result[0]
    preferred_lang = session_result[1]   # user\'s confirmed language from last turn

    # -- Step 1: STT with stored language as primary -----------------------
    # Using preferred_lang as primary tells Azure to produce native script
    # (e.g. Telugu, Punjabi Gurmukhi) instead of romanized when confidence is low.
    native_query, lang_code = _azure_stt(audio_bytes, content_type, preferred_lang)
    print(f"[STT] lang={lang_code} text=\'{native_query[:80]}\'")
    if not native_query:
        send_twilio_whatsapp(from_number, to_number,
            text="Sorry, I couldn\'t understand your voice note. Please try again.")
        return

    # -- Step 2: Translate to English --------------------------------------
    english_query = translate_text(native_query, source_lang_code=lang_code, target_lang_code="en-IN")
    print(f"[translate->EN] \'{english_query[:80]}\'")

    # -- Step 3: Session context inject ------------------------------------
    rag_query = f"{last_context}\\nNew question: {english_query}" if last_context else english_query

    # -- Step 4: RAG -> English answer -------------------------------------
    english_answer = get_rag_answer(rag_query)
    print(f"[RAG] answer=\'{english_answer[:80]}\'")

    # -- PARALLEL B: session save + answer translation ---------------------
    save_thread = threading.Thread(
        target=save_session_context,
        args=(phone_id, english_query, english_answer, lang_code),
        daemon=True,
    )
    save_thread.start()

    # -- Step 5: Translate answer back -------------------------------------
    native_answer = translate_text(english_answer, source_lang_code="en-IN", target_lang_code=lang_code)
    print(f"[translate->{lang_code}] \'{native_answer[:80]}\'")

    save_thread.join(timeout=5.0)   # must complete before Lambda exits

    # -- Step 6: TTS -> send ----------------------------------------------
    try:
        audio_url = synthesize_speech(native_answer, message_sid, lang_code=lang_code)
        send_twilio_whatsapp(from_number, to_number, media_url=audio_url)
    except Exception as e:
        print(f"[TTS] failed ({e}), sending text fallback")
        send_twilio_whatsapp(from_number, to_number, text=native_answer)
'''

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("OK: main.py patched")
