import os
from dotenv import load_dotenv
from openai import OpenAI

# 读取 .env
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "funasr-nano")
PORT = int(os.getenv("PORT", 9100))

# 配置本地服务为 OpenAI base_url
client = OpenAI(
    api_key="EMPTY",  # 本地服务不需要真实 API key
    base_url=f"http://localhost:{PORT}/v1",
)

audio_file = "example/zh.wav"  # 你的音频文件

with open(audio_file, "rb") as f:
    transcript = client.audio.transcriptions.create(
        file=f,
        model=MODEL_ID,
        language="zh",
    )

print("Transcription Result:")
print(transcript.text)

