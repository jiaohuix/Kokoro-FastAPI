import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from model_loader import get_model
from dotenv import load_dotenv
import asyncio

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "funasr-nano")
PORT = int(os.getenv("PORT", 9100))

SUPPORTED_LANGS = ["auto", "zh", "en", "ja"]

app = FastAPI(title="FunASR Nano OpenAI-Compatible Server")

# ---- Startup: Preload Model ----
@app.on_event("startup")
async def startup_event():
    print("Pre-loading FunASR Nano model ...")
    get_model()  # 预热模型
    print("Model loaded successfully.")

# ---- Health Check ----
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---- List Models ----
@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "funasr",
            }
        ]
    }

# ---- Audio Transcription Endpoint (Async + GPU Friendly) ----
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_ID),
    language: str = Form(default="auto"),
):
    if language not in SUPPORTED_LANGS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported language: {language}. Supported: {SUPPORTED_LANGS}"}
        )

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    m, kwargs = get_model()

    try:
        # 异步调用阻塞推理，放到线程池，不阻塞主事件循环
        res = await asyncio.to_thread(m.inference, [tmp_path], **kwargs)
        text = res[0][0]["text"]
        return {"text": text, "model": model, "language": language}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(tmp_path)


# ---- Main Entry ----
if __name__ == "__main__":
    import uvicorn
    print(f"Starting FunASR Nano server on port {PORT} ...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

