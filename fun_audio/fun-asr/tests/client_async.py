import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID", "funasr-nano")
PORT = int(os.getenv("PORT", 9100))

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{PORT}/v1",
)

async def transcribe_file(audio_path: str):
    """异步调用音频转写接口（内部同步函数放到线程）"""
    def sync_call():
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                file=f,
                model=MODEL_ID,
                language="zh"
            )

    transcript = await asyncio.to_thread(sync_call)
    return transcript.text

async def main():
    audio_file = "example/zh.wav"
    text = await transcribe_file(audio_file)
    print(f"Transcription Result for {audio_file}:")
    print(text)

if __name__ == "__main__":
    asyncio.run(main())

