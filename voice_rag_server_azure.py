"""voice_rag_server_azure.py — FastAPI server for Twilio webhook.

Replaces AWS Lambda with a standard FastAPI app suitable for deployment
on Azure Container Apps. 

Flow:
1. Receive Twilio POST webhook
2. Extract message, phone number, media
3. Enqueue background task for RAG/TTS processing
4. Return 200 OK immediately
"""

import time
import os
import re

from fastapi import FastAPI, Request, BackgroundTasks, Response
from fastapi.responses import PlainTextResponse

from config import blob_service_client, CONTAINER_VOICE_INPUT, CONTAINER_SESSIONS
from stt import azure_stt, download_twilio_audio, detect_lang_from_script
from tts import synthesize_speech
from translation import translate_text
from rag_logic import get_rag_answer, has_scheme_intent, extract_urls, strip_urls
from session import get_session_context, save_session_context
from twilio_utils import send_whatsapp
from greetings import _LANG_MENU, _LANG_SELECTION, _LANG_CONFIRM

app = FastAPI(title="Gov Schemes Voice RAG - Azure")

# Intent and language logic is extracted from main.py
def _handle_voice_async(
    media_url: str,
    message_sid: str,
    from_number: str,
    to_number: str,
    phone_id: str,
):
    """Full voice processing pipeline running as a Background Task."""
    print(f"Starting voice pipeline for {phone_id}")

    # 1. Load session (sync for simplicity in background task)
    ctx, preferred_lang = get_session_context(phone_id)

    # 2. Download audio
    try:
        audio_bytes, content_type = download_twilio_audio(media_url)
    except Exception as e:
        print(f"[download] failed: {e}")
        send_whatsapp(
            from_number,
            to_number,
            text=(
                "Sorry, I couldn't access your voice note from WhatsApp. "
                "Please resend the audio, or send your question as text."
            ),
        )
        return
        
    try:
        if blob_service_client:
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_VOICE_INPUT, blob=f"wa-in/{message_sid}.ogg")
            blob_client.upload_blob(audio_bytes, overwrite=True)
    except Exception as e:
        print(f"Failed to save raw audio: {e}")

    # 3. STT
    native_query, lang_code, stt_err = azure_stt(
        audio_bytes, content_type, preferred_lang=preferred_lang
    )
    if stt_err:
        print(f"[STT Error] {stt_err}")
    print(f"[STT] lang={lang_code} text='{native_query[:80]}'")

    if not native_query:
        send_whatsapp(
            from_number, to_number,
            text="Sorry, I couldn't understand your voice note. Please try again.",
        )
        return

    # 4. Translate native -> English
    english_query = translate_text(
        native_query, source_lang_code=lang_code, target_lang_code="en-IN"
    )
    print(f"[translate->EN] '{english_query[:80]}'")

    # 5. Intent Guard
    if not has_scheme_intent(english_query):
        print(f"[intent] No scheme intent in: '{english_query[:80]}'")
        clarify_en = (
            "I didn't catch a specific question about a government scheme. "
            "Could you ask something like: 'What scholarships are available "
            "for SC students?' or 'How do I apply for PM Kisan Yojana?'"
        )
        send_whatsapp(
            from_number, to_number,
            text=translate_text(
                clarify_en, source_lang_code="en-IN", target_lang_code=lang_code
            ),
        )
        return

    # 6. RAG (Local Pipeline + Azure Blob cache)
    english_answer = get_rag_answer(english_query)
    print(f"[RAG] answer='{english_answer[:80]}'")

    # 7. Save session
    save_session_context(phone_id, english_query, english_answer, lang_code)

    # 8. Translate English -> native
    native_answer = translate_text(
        english_answer, source_lang_code="en-IN", target_lang_code=lang_code
    )
    print(f"[translate->{lang_code}] '{native_answer[:80]}'")

    links    = extract_urls(english_answer)
    tts_text = strip_urls(native_answer)

    # 9. Send TEXT answer
    send_whatsapp(from_number, to_number, text=tts_text)
    print("[text-first] text answer sent, starting TTS")

    # 10. TTS -> Azure Blob -> Send Audio
    try:
        audio_url = synthesize_speech(tts_text, message_sid, lang_code=lang_code)
        if audio_url:
            send_whatsapp(from_number, to_number, media_url=audio_url)
    except Exception as e:
        print(f"[TTS] failed ({e}), text was already sent above")

    # 11. Send Links
    if links:
        link_msg = "🔗 Official link(s):\n" + "\n".join(links)
        print(f"[links] sending {len(links)} link(s)")
        send_whatsapp(from_number, to_number, text=link_msg)

def _twiml_reply_text(message: str) -> PlainTextResponse:
    twiml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{message}</Message></Response>"
    return PlainTextResponse(content=twiml, media_type="text/xml")


@app.post("/webhook")
async def twilio_webhook(request: Request, background_tasks: BackgroundTasks):
    """Twilio WhatsApp Webhook endpoint."""
    form_data = await request.form()
    params = dict(form_data)
    
    message_sid = params.get("MessageSid", f"msg-{int(time.time())}")
    raw_phone   = params.get("From", "unknown")
    phone_id    = raw_phone.replace("+", "").replace(":", " ")
    media_url   = params.get("MediaUrl0")
    user_text   = params.get("Body", "").strip()

    # Voice Note
    if media_url:
        print(f"Processing Voice Note from: {media_url}")
        background_tasks.add_task(
            _handle_voice_async,
            media_url=media_url,
            message_sid=message_sid,
            from_number=params.get("From", ""),
            to_number=params.get("To", ""),
            phone_id=phone_id
        )
        return _twiml_reply_text("Got your voice note! Searching for schemes, reply in a moment...")

    # Sandbox join / greeting
    if user_text.lower().startswith("join ") or user_text.strip().lower() in (
        "hi", "hello", "start", "help", "language", "change language"
    ):
        return _twiml_reply_text(_LANG_MENU)

    # Language selection
    _sel_key = user_text.strip().lower()
    if _sel_key in _LANG_SELECTION:
        chosen_lang = _LANG_SELECTION[_sel_key]
        if blob_service_client:
            try:
                blob_client = blob_service_client.get_blob_client(container=CONTAINER_SESSIONS, blob=f"{phone_id}.txt")
                blob_client.upload_blob(f"LANG:{chosen_lang}\n".encode("utf-8"), overwrite=True)
            except Exception as e:
                print(f"[lang-select] failed to save to blob: {e}")
                
        print(f"[lang-select] {phone_id} chose {chosen_lang}")
        return _twiml_reply_text(_LANG_CONFIRM[chosen_lang])

    # Text query fallback
    if not user_text:
        return _twiml_reply_text(_LANG_MENU)

    lang_code = detect_lang_from_script(user_text, fallback="en-IN")
    print(f"[text] lang={lang_code} query='{user_text[:80]}'")

    english_query = (
        translate_text(user_text, source_lang_code=lang_code, target_lang_code="en-IN")
        if lang_code != "en-IN" else user_text
    )

    english_ans = get_rag_answer(english_query)

    native_ans = (
        translate_text(english_ans, source_lang_code="en-IN", target_lang_code=lang_code)
        if lang_code != "en-IN" else english_ans
    )

    links = extract_urls(english_ans)
    reply = strip_urls(native_ans)
    if links:
        reply += "\n\n🔗 Official link(s):\n" + "\n".join(links)

    save_session_context(phone_id, english_query, english_ans, lang_code)
    return _twiml_reply_text(reply)

@app.get("/health")
def health_check():
    return {"status": "ok"}
