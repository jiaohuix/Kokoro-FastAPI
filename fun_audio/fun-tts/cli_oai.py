from openai import OpenAI

# 指向你的本地 FastAPI 地址
client = OpenAI(api_key="sk-123", base_url="http://127.0.0.1:9102/v1")

with client.audio.speech.with_streaming_response.create(
    model="cosyvoice",
    # voice="hemi",
    voice="female-shaonv",
    input="人生在世，谁不曾有过困顿失意之时？恰似行船遇逆风，驱车登陡坡，进退维谷，举步维艰。",
    response_format="wav", # 或者 wav
) as response:
    response.stream_to_file("output_test3.mp3")