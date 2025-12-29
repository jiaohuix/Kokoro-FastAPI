import os
import json
import requests
import gradio as gr
import numpy as np
import scipy.io.wavfile as wavfile

# ================= 配置区 =================
TTS_SERVER_URL = "http://127.0.0.1:9102/tts"

INSTRUCT_LIST = [
    "You are a helpful assistant. 请用上海话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用东北话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用甘肃话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用贵州话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用河南话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用湖北话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用湖南话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用江西话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用闽南话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用宁夏话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用山西话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用陕西话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用山东话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用四川话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用天津话表达。<|endofprompt|>",
    "You are a helpful assistant. 请用云南话表达。<|endofprompt|>",
    "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
    "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
    "You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>",
    "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
    "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
    "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
    "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
    "You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>",
    "You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>"
]

# ================= 核心调用逻辑 =================
def call_tts_api(tts_text, mode_choice, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, speed, is_stream):
    # 1. 确定音频来源
    wav_path = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    
    # 2. 基础校验
    if not wav_path:
        raise gr.Error("请提供参考音频！")
    
    api_mode = "zero_shot" if mode_choice == "3s极速复刻" else "instruct"
    
    if api_mode == "zero_shot" and not prompt_text.strip():
        raise gr.Error("3s极速复刻模式必须提供【参考文本】（音频里的原话）")

    # 3. 构造 Payload
    payload = {
        "tts_text": tts_text,
        "mode": api_mode,
        "prompt_text": prompt_text if api_mode == "zero_shot" else "",
        "instruct_text": instruct_text if api_mode == "instruct" else "",
        "seed": int(seed),
        "speed": float(speed),
        "stream": str(is_stream).lower()
    }

    try:
        with open(wav_path, "rb") as f:
            files = {"prompt_wav": (os.path.basename(wav_path), f, "audio/wav")}
            # 增加 timeout，模型初始化可能较慢
            response = requests.post(TTS_SERVER_URL, data=payload, files=files, stream=is_stream, timeout=120)
        
        if response.status_code != 200:
            try:
                err_msg = response.json().get("error", response.text)
            except:
                err_msg = response.text
            raise gr.Error(f"服务报错: {err_msg}")

        # 4. 处理返回数据
        full_audio = []
        if is_stream:
            # 流式处理
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if "error" in data: raise gr.Error(data["error"])
                    full_audio.extend(data["audio"])
        else:
            # 非流式处理
            res = response.json()
            if isinstance(res, dict) and "error" in res:
                raise gr.Error(res["error"])
            full_audio = res["audio"]

        # 5. 音频后处理：转为 int16 适配 Gradio
        audio_np = np.array(full_audio, dtype=np.float32)
        # 简单归一化防止爆音
        if np.abs(audio_np).max() > 0:
            audio_np = audio_np / np.abs(audio_np).max() * 0.9
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return (24000, audio_int16)

    except Exception as e:
        if isinstance(e, gr.Error): raise e
        raise gr.Error(f"调用失败: {str(e)}")

# ================= UI 界面 =================
def main():
    with gr.Blocks(title="CosyVoice Pro v5") as demo:
        gr.Markdown("### 🔊 CosyVoice API 语音合成控制台")
        
        with gr.Row():
            with gr.Column(scale=1):
                tts_text = gr.Textbox(label="1. 输入合成正文", lines=3, value="你好啊我是你的健康助手安安，有任何健康相关的疑问都可以来问我哦。")
                mode_choice = gr.Radio(["3s极速复刻", "自然语言控制"], label="2. 推理模式", value="3s极速复刻")
                
                with gr.Group():
                    gr.Markdown("#### 3. 音色参考")
                    prompt_audio_up = gr.Audio(sources='upload', type='filepath', label="上传音频 (15s内)")
                    prompt_audio_rec = gr.Audio(sources='microphone', type='filepath', label="或者直接录音")
                    prompt_text = gr.Textbox(label="参考文本 (Prompt Text)", placeholder="输入参考音频里的原话，3s模式必填")

                with gr.Group(visible=False) as instruct_panel:
                    gr.Markdown("#### 4. 风格指令")
                    instruct_text = gr.Dropdown(choices=INSTRUCT_LIST, label="指令列表 (可自行输入)", allow_custom_value=True)

                with gr.Row():
                    seed = gr.Number(value=42, label="随机种子", precision=0)
                    speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速")
                    is_stream = gr.Checkbox(label="开启流式传输", value=False)
                
                btn = gr.Button("🚀 立即开始合成", variant="primary")

            with gr.Column(scale=1):
                audio_out = gr.Audio(label="合成结果", interactive=False, autoplay=True)
                gr.Info("提示：3s模式侧重于音色还原，Instruct模式侧重于情感/方言控制。")

        # 交互逻辑：切换模式时显示/隐藏指令面板
        mode_choice.change(
            fn=lambda m: gr.update(visible=(m=="自然语言控制")), 
            inputs=[mode_choice], 
            outputs=[instruct_panel]
        )
        
        btn.click(
            fn=call_tts_api, 
            inputs=[tts_text, mode_choice, prompt_text, prompt_audio_up, prompt_audio_rec, instruct_text, seed, speed, is_stream], 
            outputs=[audio_out]
        )

    demo.launch(server_name="0.0.0.0", server_port=9103)

if __name__ == "__main__":
    main()