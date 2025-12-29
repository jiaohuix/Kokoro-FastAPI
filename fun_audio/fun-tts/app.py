import os
import sys
import uuid
import json
import torch
import random
import regex
import librosa
import logging
import torchaudio
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Optional,List, Dict, Union
import time
from datetime import datetime
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
import uuid, json, numpy as np, os, subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor

debug = False

# 创建一个全局线程池用于处理同步的模型推理
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="Audio Platform API", version="1.0")

# --- 基础配置 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceAutoModel
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav

class AudioVoiceConsent(BaseModel):
    object: str = "audio.voice_consent"
    id: str
    name: str
    language: Optional[str] = None
    created_at: int

class ListResponse(BaseModel):
    object: str = "list"
    data: List[AudioVoiceConsent]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool




# 配置日志
logging.basicConfig(level=logging.INFO)

# 从环境变量获取配置，提供默认值作为后备
TENANTS_ROOT = os.getenv("TENANTS_ROOT", "/workspace/audio_data/tenants/")
TENANT_ID = os.getenv("TENANT_ID", "tenant_system")

# --- 模型初始化 ---
MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models/CosyVoice3-0.5B-2512-modelscope")
logging.info(f"Loading model from: {MODEL_DIR}")
logging.info(f"Tenants root: {TENANTS_ROOT}")

# 建议：如果显存允许，使用 fp16=True 提升速度
cosyvoice = CosyVoiceAutoModel(model_dir=MODEL_DIR, load_trt=False, fp16=False)

target_sr = 24000
max_val = 0.8

# --- 核心工具函数 ---
def count_chars_and_words(text):
    if not text: return 0
    punctuation_pattern = r'[\p{P}\p{S}\s]+'
    text_no_punct = regex.sub(punctuation_pattern, ' ', text)
    pattern = r'\p{Han}|\p{Latin}+'
    matches = regex.findall(pattern, text_no_punct, regex.UNICODE)
    return len(matches)

def postprocess_wav(wav_path):
    """鲁棒性处理：确保音频加载失败不崩溃"""
    try:
        speech = load_wav(wav_path, target_sr=target_sr, min_sr=16000)
        # trim 处理
        speech_trimmed, _ = librosa.effects.trim(
            speech.numpy(), top_db=60, frame_length=440, hop_length=220
        )
        speech = torch.from_numpy(speech_trimmed)
        # 归一化
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        # 补 0.2s 静音防止截断
        speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
        torchaudio.save(wav_path, speech, target_sr)
        return True
    except Exception as e:
        logging.error(f"Postprocess error: {e}")
        return False

# --- 增强型生成器 ---
def generate_tts_chunks(tts_text, mode, prompt_text, instruct_text, tmp_path, seed, speed, stream):
    """
    鲁棒性增强：
    1. 确保 tmp_path 存在且有效
    2. 针对 0.5B 系列逻辑分支优化
    3. 完善的错误捕获
    """
    try:
        set_all_random_seed(seed)
        
        # 模式校验与逻辑分发
        if mode == 'zero_shot':
            if not tmp_path or not os.path.exists(tmp_path):
                yield json.dumps({"error": "3s极速复刻必须上传参考音频"}) + "\n"
                return
            # 对齐源码：前缀必须固定
            full_prompt_text = 'You are a helpful assistant.<|endofprompt|>' + (prompt_text or "")
            model_output = cosyvoice.inference_zero_shot(tts_text, full_prompt_text, tmp_path, stream=stream, speed=speed)
        
        elif mode == 'instruct':
            if not instruct_text:
                yield json.dumps({"error": "自然语言控制模式必须提供指令文本"}) + "\n"
                return
            # 如果没有音频，部分版本会报错，这里强制要求
            if not tmp_path or not os.path.exists(tmp_path):
                yield json.dumps({"error": "Instruct模式也需要参考音频来确定音色"}) + "\n"
                return
            model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, tmp_path, stream=stream, speed=speed)
        
        else:
            yield json.dumps({"error": f"不支持的模式: {mode}"}) + "\n"
            return

        # 迭代输出
        for res in model_output:
            if 'tts_speech' in res:
                audio_data = res['tts_speech'].numpy().flatten().tolist()
                yield json.dumps({"sample_rate": target_sr, "audio": audio_data}) + "\n"

    except Exception as e:
        logging.error(f"Inference error detail: {str(e)}")
        yield json.dumps({"error": f"模型推理异常: {str(e)}"}) + "\n"
    
    finally:
        # 生成器结束或异常时，确保清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logging.info(f"Cleaned tmp file: {tmp_path}")
            except:
                pass


# --- FastAPI 接口 ---
app = FastAPI(title="Audio  API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def load_voice_and_sample(voice_id: str):
    voice_dir = os.path.join(TENANTS_ROOT, TENANT_ID, "default", "voices", voice_id)
    if not os.path.isdir(voice_dir):
        return None, None

    samples_dir = os.path.join(voice_dir, "samples")
    if not os.path.isdir(samples_dir):
        return None, None

    chosen = None
    for s in sorted(os.listdir(samples_dir)):
        p = os.path.join(samples_dir, s, "sample.json")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("is_default_reference"):
            chosen = data
            break
        if chosen is None:
            chosen = data

    if not chosen:
        return None, None

    audio_path = os.path.join(
        voice_dir, "samples", chosen["sample_id"], "audio.wav"
    )
    return chosen["text"], audio_path

class VoiceResolver:
    """
    基于本地 voice 管理目录解析 voice_id。
    返回参考音频路径 + 默认文本
    """

    @staticmethod
    def resolve(voice_id: str):
        text, wav = load_voice_and_sample(voice_id)
        if text is None or wav is None:
            raise ValueError(f"未找到音色 {voice_id}")
        return {"ref_text": text, "ref_wav": wav}



## list  voices

def iso_to_unix(ts: str) -> int:
    """
    2025-12-25T08:00:00Z -> unix seconds
    """
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "")).timestamp())
    except Exception:
        return int(time.time())


def list_voice_consents(
    limit: int = 20,
    after: Optional[str] = None,
) -> dict:

    voices_path = os.path.join(TENANTS_ROOT, TENANT_ID, "default", "voices")
    
    if not os.path.exists(voices_path):
        return ListResponse(
            data=[],
            first_id=None,
            last_id=None,
            has_more=False,
        ).model_dump(exclude_none=True)

    items: List[AudioVoiceConsent] = []


    for voice_id in sorted(os.listdir(voices_path)):
        voice_json = os.path.join(voices_path, voice_id, "voice.json")
        if not os.path.isfile(voice_json):
            continue

        with open(voice_json, "r", encoding="utf-8") as f:
            v = json.load(f)

        language = v.get("language")
        if isinstance(language, list):
            language = language[0]

        items.append(
            AudioVoiceConsent(
                id=v.get("voice_id", voice_id),
                name=v.get("display_name", voice_id),
                language=language,
                created_at=iso_to_unix(v.get("created_at")),
            )
        )

    # -------- pagination --------
    start = 0
    if after:
        for i, it in enumerate(items):
            if it.id == after:
                start = i + 1
                break

    page = items[start:start + limit]
    has_more = start + limit < len(items)

    resp = ListResponse(
        data=page,
        first_id=page[0].id if page else None,
        last_id=page[-1].id if page else None,
        has_more=has_more,
    )
    
    return resp.model_dump(exclude_none=True)

@app.get("/v1/audio/voice_consents")
def api_list_voice_consents(
    limit: int = Query(20, ge=1, le=100),
    after: Optional[str] = Query(None),
):
    return list_voice_consents(limit=limit, after=after)


@app.post("/tts")
async def tts_endpoint(
    tts_text: str = Form(...),
    mode: str = Form(...),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    seed: int = Form(0),
    speed: float = Form(1.0),
    stream: bool = Form(False),
    prompt_wav: UploadFile = File(None)
):
    # --- 新增打印部分 ---
    print("\n" + "="*30)
    print(f"收到请求 | 模式: {mode}")
    print(f"合成文本: {tts_text}")
    print(f"参考文本: {prompt_text}")
    print(f"指令文本: {instruct_text}")
    print(f"随机种子: {seed} | 语速: {speed} | 流式: {stream}")
    if prompt_wav:
        print(f"参考音频: {prompt_wav.filename}")
    print("="*30 + "\n")
    # -------------------

    # 1. 基础合法性检查
    if not tts_text.strip():
        return {"error": "待合成文本不能为空"}
    # ... 后面保持不变
    
    if count_chars_and_words(tts_text) > 200:
        return {"error": "文本超过200字限制"}

    # 2. 唯一化文件名处理并发
    task_id = str(uuid.uuid4())
    tmp_path = f"tmp_{task_id}.wav"

    try:
        # 3. 处理参考音频
        if prompt_wav:
            content = await prompt_wav.read()
            if not content:
                return {"error": "上传的音频文件为空"}
                
            with open(tmp_path, "wb") as f:
                f.write(content)
            
            # 音频校验
            info = torchaudio.info(tmp_path)
            if info.num_frames / info.sample_rate > 15:
                if os.path.exists(tmp_path): os.remove(tmp_path)
                return {"error": "参考音频不可超过15秒"}
            
            if not postprocess_wav(tmp_path):
                if os.path.exists(tmp_path): os.remove(tmp_path)
                return {"error": "音频预处理失败，请检查文件格式"}
        else:
            tmp_path = None

        # 4. 返回流或结果
        if stream:
            return StreamingResponse(
                generate_tts_chunks(tts_text, mode, prompt_text, instruct_text, tmp_path, seed, speed, stream),
                media_type="application/x-ndjson"
            )
        else:
            full_audio = []
            # 手动迭代生成器
            for line in generate_tts_chunks(tts_text, mode, prompt_text, instruct_text, tmp_path, seed, speed, stream):
                item = json.loads(line)
                if "error" in item:
                    return item
                full_audio.extend(item.get("audio", []))
            
            return {"sample_rate": target_sr, "audio": full_audio}

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
        logging.error(f"Endpoint error: {e}")
        return {"error": f"系统错误: {str(e)}"}



# ------------------- Pydantic 请求模型 -------------------
import base64
import struct
from fastapi import BackgroundTasks
from pydantic import Field

# ------------------- OpenAI 兼容请求模型 -------------------
class OpenAISpeechRequest(BaseModel):
    model: str
    input: str = Field(..., max_length=4096)
    voice: Union[str, Dict]
    instructions: Optional[str] = ""
    # emotion: str = ""
    response_format: str = "mp3"  # mp3, opus, aac, flac, wav, pcm
    speed: float = Field(1.0, ge=0.25, le=4.0)
    stream: bool = False
    stream_format: str = "audio"  # "audio" 或 "sse"


async def generate_cosy_audio_stream(tts_text, mode, ref_text, instruct_text, ref_wav_path, speed):
    """
    安全且带日志的 CosyVoice TTS 异步流生成器
    - 自动兼容 float64、float32、Tensor、list、tuple、标量
    - 转为 int16 PCM 字节流
    - 异常捕获并打印详细日志
    """
    try:
        logging.info(f"[TTS Stream] Start generation | mode={mode} | speed={speed}")
        
        # 设置随机种子
        set_all_random_seed(0)

        # 推理函数
        def run_inference():
            logging.info("[TTS Stream] Running model inference...")
            if mode == 'zero_shot':
                full_prompt_text = 'You are a helpful assistant.<|endofprompt|>' + (ref_text or "")
                return cosyvoice.inference_zero_shot(tts_text, full_prompt_text, ref_wav_path, stream=True, speed=speed)
            else:
                return cosyvoice.inference_instruct2(tts_text, instruct_text, ref_wav_path, stream=True, speed=speed)

        # 在线程池中执行同步推理
        loop = asyncio.get_event_loop()
        model_output = await loop.run_in_executor(executor, run_inference)

        for idx, res in enumerate(model_output):
            if 'tts_speech' not in res:
                logging.warning(f"[TTS Stream] Chunk {idx} missing 'tts_speech'")
                continue

            audio_tensor = res['tts_speech']

            # 1. 转为 numpy ndarray
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.detach().cpu().numpy()
                if debug:
                    logging.debug(f"[TTS Stream] Chunk {idx} type: torch.Tensor shape={audio_np.shape}")
            elif isinstance(audio_tensor, np.ndarray):
                audio_np = audio_tensor
                if debug:
                    logging.debug(f"[TTS Stream] Chunk {idx} type: np.ndarray shape={audio_np.shape}")
            elif isinstance(audio_tensor, (list, tuple)):
                audio_np = np.array(audio_tensor, dtype=np.float32)
                if debug:
                    logging.debug(f"[TTS Stream] Chunk {idx} type: list/tuple length={len(audio_tensor)}")
            else:
                audio_np = np.array([float(audio_tensor)], dtype=np.float32)
                if debug:
                    logging.debug(f"[TTS Stream] Chunk {idx} type: scalar")

            # 2. flatten 并 clip
            audio_np = audio_np.flatten()
            audio_np = np.clip(audio_np, -1.0, 1.0)

            # 3. 转 int16
            audio_int16 = (audio_np * 32767.0).astype(np.int16)

            logging.info(f"[TTS Stream] Yielding chunk {idx} | samples={len(audio_int16)}")
            yield audio_int16.tobytes()

            # 给事件循环喘息
            await asyncio.sleep(0)

        logging.info("[TTS Stream] Generation complete.")

    except Exception as e:
        logging.error(f"[TTS Stream] Exception: {e}")


# SSE 包装器
async def wrap_done_event():
    """构造结束事件"""
    done_data = {
        "type": "speech.audio.done",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }
    return f"data: {json.dumps(done_data)}\n\n"

async def openai_sse_formatter(binary_generator):
    """将二进制流转化为 OpenAI SSE 格式"""
    async for chunk in binary_generator:
        b64_audio = base64.b64encode(chunk).decode('utf-8')
        event_data = {
            "type": "speech.audio.delta",
            "audio": b64_audio
        }
        yield f"data: {json.dumps(event_data)}\n\n"
    
    yield await wrap_done_event()

# 路由接口实现
@app.post("/v1/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    background_tasks: BackgroundTasks
):
    
    """OpenAI-compatible TTS endpoint"""
    # 检查输入
    if not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request_error",
                "message": "input不能为空",
                "type": "invalid_request_error"
            }
        )

    # 增加模型适配逻辑（可选）
    if request.model not in ["cosyvoice", "speech-01"]:
        logging.warning(f"Unsupported model requested: {request.model}, defaulting to cosyvoice")

    # 1. 解析 Voice ID
    if isinstance(request.voice, str):
        voice_id = request.voice
    elif isinstance(request.voice, dict): # 只能传str吧
        voice_id = request.voice.get("id")
    else:
        raise HTTPException(status_code=400, detail="voice格式错误")

    try:
        ref_info = VoiceResolver.resolve(voice_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if debug:
        print("ref_info",ref_info)
    # 2. 准备参数
    # mode = "instruct" if request.instructions else "zero_shot"
    mode = "zero_shot"

    media_type_map = {
        "mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm",
        "opus": "audio/opus", "aac": "audio/aac", "flac": "audio/flac"
    }
    content_type = media_type_map.get(request.response_format, "audio/mpeg")

    # 定义异步二进制流
    binary_gen = generate_cosy_audio_stream(
        tts_text=request.input,
        mode=mode,
        ref_text=ref_info["ref_text"],
        instruct_text=request.instructions,
        ref_wav_path=ref_info["ref_wav"],
        speed=request.speed
    )
    print("binary_gen",binary_gen)

    # 3. 处理流式返回
    if request.stream:
        if request.stream_format == "sse":
            return StreamingResponse(openai_sse_formatter(binary_gen), media_type="text/event-stream")
        else:
            # 注意：如果是请求 mp3 但又是二进制流，这里直接输出 PCM 字节客户端通常也能处理（wav 封装）
            # 严格来说流式转码 mp3 需要 ffmpeg pipe，这里先返回原始流
            return StreamingResponse(binary_gen, media_type=content_type)

    # 4. 处理非流式返回 (生成完整文件)
    else:
        print("stream false")
        task_id = str(uuid.uuid4())
        base_path = f"./tmp/{task_id}"
        os.makedirs("./tmp", exist_ok=True)
        pcm_path = f"{base_path}.pcm"
        final_path = f"{base_path}.{request.response_format}"

        # 收集所有块
        all_bytes = b""
        async for chunk in binary_gen:
            all_bytes += chunk
        
        if not all_bytes:
            raise HTTPException(status_code=500, detail="Audio generation failed")

        # 写入临时 PCM 
        print("写入pcm", pcm_path)
        with open(pcm_path, "wb") as f:
            f.write(all_bytes)

        # 格式转换
        if request.response_format in ["pcm", "wav"]:
            if request.response_format == "wav":
                import soundfile as sf
                audio_np = np.frombuffer(all_bytes, dtype=np.int16)
                sf.write(final_path, audio_np, target_sr)
                output_file = final_path
            else:
                output_file = pcm_path
        else:
            # 使用 ffmpeg 转码为 mp3/aac/etc.
            # -f s16le: 输入为 16bit 采样, -ar 24000: 采样率 24k, -ac 1: 单声道
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-f", "s16le", "-ar", str(target_sr), "-ac", "1", 
                    "-i", pcm_path, final_path
                ], check=True, capture_output=True)
                output_file = final_path
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error: {e.stderr.decode()}")
                output_file = pcm_path # 退回到原文件

        # 注册清理任务：传输完成后删除
        background_tasks.add_task(lambda: [os.remove(f) for f in [pcm_path, final_path] if os.path.exists(f)])

        return FileResponse(output_file, media_type=content_type)
    



if __name__ == "__main__":
    port = int(os.getenv("PORT", 9102))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)