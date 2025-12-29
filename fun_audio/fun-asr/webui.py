import os
import gradio as gr
import asyncio
from openai import OpenAI
import tempfile
import requests

# ==============================
# 配置本地 ASR 服务参数
# ==============================
MODEL_ID = os.getenv("MODEL_ID", "funasr-nano")  # 模型ID，可通过.env修改
PORT = int(os.getenv("PORT", 9100))             # 服务端口

client = OpenAI(
    api_key="EMPTY", 
    base_url=f"http://localhost:{PORT}/v1"
)

# ==============================
# 异步转写函数
# ==============================
async def transcribe_audio(audio_file, audio_url):
    """
    调用本地 ASR 服务进行音频转写
    支持上传文件或输入音频 URL
    """
    try:
        # 确定音频来源
        audio_path = audio_file
        if not audio_path and audio_url:
            r = requests.get(audio_url, stream=True)
            if r.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                    audio_path = f.name
            else:
                return f"下载音频失败，状态码：{r.status_code}", "", None

        if not audio_path:
            return "请上传音频文件或输入音频 URL", "", None

        # 调用同步 SDK 放到线程池异步执行
        def sync_transcribe():
            with open(audio_path, "rb") as f:
                res = client.audio.transcriptions.create(
                    file=f,
                    model=MODEL_ID,
                    language="zh"
                )
            return res.text

        transcription = await asyncio.to_thread(sync_transcribe)

        # 保存转写结果到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tf:
            tf.write(transcription)
            transcription_file = tf.name

        return "转写完成", transcription, transcription_file

    except Exception as e:
        return f"转写出错：{str(e)}", "", None

# ==============================
# Gradio WebUI
# ==============================
with gr.Blocks() as iface:
    gr.Markdown("# 本地语音转写服务 🎤")
    gr.Markdown("上传音频或输入音频 URL，即可进行中文语音转写")

    with gr.Row():
        audio_input = gr.Audio(label="上传音频", type="filepath", sources=["upload", "microphone"])
        audio_url = gr.Textbox(label="音频 URL", placeholder="直接链接到 wav/mp3 文件")

    transcribe_btn = gr.Button("开始转写")

    with gr.Row():
        status_output = gr.Textbox(label="状态", lines=2)
        transcription_output = gr.Textbox(label="转写文本", lines=10)
        transcription_file = gr.File(label="下载转写结果")

    # 按钮回调
    async def submit(audio_file, audio_url):
        status, transcription, file_path = await transcribe_audio(audio_file, audio_url)
        return status, transcription, file_path

    transcribe_btn.click(
        submit, 
        inputs=[audio_input, audio_url], 
        outputs=[status_output, transcription_output, transcription_file]
    )

    gr.Markdown("""
    ### 使用说明：
    1. 上传音频或输入音频 URL（wav/mp3 格式）。
    2. 点击 **开始转写**，等待结果。
    3. 可下载生成的转写文本。
    """)

# ==============================
# 启动 WebUI
# ==============================
iface.queue().launch(server_name="0.0.0.0",server_port=9101, debug=True, share=True)

