import requests
import json
import numpy as np
import scipy.io.wavfile as wavfile
import os

# 配置服务端地址
URL = "http://127.0.0.1:9102/tts"

def test_cosyvoice(mode, tts_text, prompt_wav_path, prompt_text="", instruct_text="", output_name="output.wav"):
    print(f"\n>>> 正在测试模式: {mode}")
    
    # 确保参考音频存在
    if not os.path.exists(prompt_wav_path):
        print(f"错误: 找不到参考音频文件 {prompt_wav_path}")
        return

    # 构造表单数据
    # 注意：根据你的服务端代码，mode 接收 'zero_shot' 或 'instruct'
    data = {
        "tts_text": tts_text,
        "mode": mode,
        "prompt_text": prompt_text,
        "instruct_text": instruct_text,
        "seed": 42,
        "speed": 1.0,
        "stream": "false"  # 确保不使用流式，方便 CLI 演示
    }

    # 构造文件数据
    files = {
        "prompt_wav": open(prompt_wav_path, "rb")
    }

    try:
        response = requests.post(URL, data=data, files=files, timeout=120)
        
        if response.status_code == 200:
            res_json = response.json()
            if "error" in res_json:
                print(f"服务端逻辑错误: {res_json['error']}")
            else:
                # 获取音频数据并转回 numpy
                audio_data = np.array(res_json["audio"], dtype=np.float32)
                sr = res_json["sample_rate"]
                
                # 保存为 wav 文件
                wavfile.write(output_name, sr, audio_data)
                print(f"成功！音频已保存至: {output_name}")
        else:
            print(f"HTTP 请求失败，状态码: {response.status_code}")
            print(f"详情: {response.text}")
            
    except Exception as e:
        print(f"调用异常: {str(e)}")
    finally:
        files["prompt_wav"].close()

if __name__ == "__main__":
    # 测试参数
    # REF_WAV = "examples/reference_male.wav"
    # REF_TEXT = "主人公必须在理想和责任之间做出一个艰难的选择。"
    REF_WAV = "examples/reference_female.wav"
    REF_TEXT = "谢谢你的帮助，你人真好。"

    CONTENT = "你好啊我是你的健康助手安安，有任何健康相关的疑问都可以来问我哦。"

    # 1. 测试 3s极速复刻 (zero_shot)
    # 注意：此模式下需要提供参考音频里“原话”的内容作为 prompt_text
    test_cosyvoice(
        mode="zero_shot",
        tts_text=CONTENT,
        prompt_wav_path=REF_WAV,
        prompt_text=REF_TEXT, # 假设这是 reference_male.wav 里的原话
        output_name="test_zero_shot.wav"
    )

    # 2. 测试 自然语言控制 (instruct)
    # 注意：此模式下 prompt_text 会被忽略，重点是 instruct_text
    test_cosyvoice(
        mode="instruct",
        tts_text=CONTENT,
        prompt_wav_path=REF_WAV,
        instruct_text="You are a helpful assistant. 请用上海话表达。<|endofprompt|>",
        output_name="test_instruct.wav"
    )