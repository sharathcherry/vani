## Required components to include

### User & Ingestion
- WhatsApp (voice note + text) via Twilio Business API
- Twilio webhook → API Gateway / Lambda URL → `vani-jan-webhook`

### Lambda: vani-jan-webhook (Zip-deployed)
- Sync `lambda_handler` – returns TwiML ACK in <15 s (Twilio hard limit)
- Async `_handle_voice_async` – self-invoked fire-and-forget
- Parallel block A: Azure Speech STT (Language ID, `preferred_lang` from session) ‖ S3 session read
- ffmpeg Lambda Layer (static binary, AAC/OGG → WAV 16 kHz) for non-OGG audio
- Bedrock Nova Lite: translate detected language → English
- `get_rag_answer`: L1 in-memory LRU (256 slots) → L2 S3 `wa-cache/` (SHA-256, 24 h TTL) → L3 RAG Lambda invoke
- Bedrock Nova Lite: translate English answer → user's detected language
- Azure Neural TTS (12 Indian language voices) → MP3 → S3 `wa-out/`
- Twilio REST API `send_twilio_whatsapp()` with presigned MP3 URL

### Lambda: gov-schemes-voice-rag (Container, ECR eu-north-1)
- `/debug/query` endpoint – pure English RAG
- fastembed: `BAAI/bge-large-en-v1.5` (dense) + `Qdrant/bm25` (sparse) – parallel threads
- Qdrant Cloud `schemes_hybrid` collection – Hybrid Search RRF (K=20)
- Azure AI Foundry Cohere `rerank-v4-fast` – Top 5 chunks
- Amazon Bedrock `eu.nova-lite-v1:0` – English answer (≤80 words, voice-optimised)

### S3 Buckets
- `whatsapp-voice-messages`: `wa-in/` (audio), `wa-sessions/` (3 h TTL session), `wa-cache/` (24 h answer cache)
- `whatsapp-voice-responses`: `wa-out/` (MP3 audio output)

### Data Layer
- ~800 Indian Govt Scheme PDFs → `bedrock_chunks/` + `structured_schemes/`
- `upsert_telangana_schemes.py` → Qdrant Cloud ingestion

## Diagram instructions
- Use `flowchart TD`
- Group components into labelled `subgraph` blocks (User, Twilio, vani-jan-webhook Lambda, gov-schemes-voice-rag Lambda, ffmpeg Layer, S3 buckets, Data Layer)
- Show parallel execution with `&` branching
- Annotate critical edges (latency, data type)
- Apply colour-coded `style` per subgraph for readability
- **Prefix every node label with the official service emoji icon:**
  - AWS Lambda
  - Amazon S3 
  - Amazon Bedrock
  - Amazon ECR
  - Qdrant Cloud 
  - Azure Speech STT
  - Azure Neural TTS 
  - Azure AI Foundry / Cohere
  - ffmpeg  
  - Twilio 
  - WhatsApp 
  - PDF / Data
- Render with `renderMermaidDiagram`


