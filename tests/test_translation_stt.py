import pytest
from unittest.mock import patch, MagicMock
from stt import detect_lang_from_script, azure_stt
from translation import translate_text

def test_detect_lang_from_script_telugu():
    # Telugu script string
    text = "నమస్కారం, నేను ఒక రైతుని"
    lang = detect_lang_from_script(text)
    assert lang == "te-IN"

def test_detect_lang_from_script_hindi():
    # Hindi script string
    text = "नमस्ते, मैं एक किसान हूँ"
    lang = detect_lang_from_script(text)
    assert lang == "hi-IN"

def test_detect_lang_from_script_fallback():
    # English string
    text = "Hello, I am a farmer"
    lang = detect_lang_from_script(text)
    assert lang == "en-IN"

@patch('stt.speechsdk.SpeechRecognizer')
@patch('stt.convert_to_wav')
def test_azure_stt_success(mock_convert, mock_recognizer):
    # Mocking convert_to_wav
    mock_convert.return_value = b'wav_bytes'

    # Mocking Azure Speech SDK behavior
    import azure.cognitiveservices.speech as speechsdk
    mock_recognize_result = MagicMock()
    mock_recognize_result.reason = speechsdk.ResultReason.RecognizedSpeech
    mock_recognize_result.text = "Hello, testing."
    
    # Need to patch AutoDetectSourceLanguageResult
    with patch('stt.speechsdk.AutoDetectSourceLanguageResult') as mock_auto_detect:
        mock_auto_result = MagicMock()
        mock_auto_result.language = "en-IN"
        mock_auto_detect.return_value = mock_auto_result
        
        mock_recognizer_instance = mock_recognizer.return_value
        mock_recognizer_instance.recognize_once_async.return_value.get.return_value = mock_recognize_result
        
        transcript, lang = azure_stt(b"audio", "audio/ogg", "te-IN")
        
        assert transcript == "Hello, testing."
        assert lang == "en-IN"
