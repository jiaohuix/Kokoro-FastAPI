"""
OpenAI 兼容 API 路由
"""
import os
import json
import uuid
import base64
import subprocess
import numpy as np
from typing import Optional, Dict, Union, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import soundfile as sf

from ..services.tts_service import TTSService

# Pydantic 模型
class OpenAISpeechRequest(BaseModel):
    model: str
    input: str = Field(..., max_length=4096)
    voice: Union[str, Dict]
    instructions: Optional[str] = ""
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    stream: bool = False
    stream_format: str = "audio"

class VoiceCreateRequest(BaseModel):
    name: str
    reference_text: str
    language: str = "zh"
    description: str = ""

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

def create_openai_router(tts_service: TTSService) -> APIRouter:
    """创建 OpenAI 兼容路由"""
    router = APIRouter()
    
    @router.post("/v1/audio/speech")
    async def create_speech(
        request: OpenAISpeechRequest,
        background_tasks: BackgroundTasks
    ):
        """OpenAI 兼容的语音合成接口"""
        
        # 输入验证
        if not request.input.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_request_error",
                    "message": "input不能为空",
                    "type": "invalid_request_error"
                }
            )
        
        # 解析 voice_id
        if isinstance(request.voice, str):
            voice_id = request.voice
        elif isinstance(request.voice, dict):
            voice_id = request.voice.get("id")
        else:
            raise HTTPException(status_code=400, detail="voice格式错误")
        
        # 媒体类型映射
        media_type_map = {
            "mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm",
            "opus": "audio/opus", "aac": "audio/aac", "flac": "audio/flac"
        }
        content_type = media_type_map.get(request.response_format, "audio/mpeg")
        
        # 生成音频流
        audio_generator = tts_service.synthesize_speech(
            text=request.input,
            voice_id=voice_id,
            instructions=request.instructions,
            speed=request.speed,
            stream=request.stream
        )
        
        # 处理流式返回
        if request.stream:
            if request.stream_format == "sse":
                return StreamingResponse(
                    _openai_sse_formatter(audio_generator),
                    media_type="text/event-stream"
                )
            else:
                return StreamingResponse(
                    _binary_audio_stream(audio_generator),
                    media_type=content_type
                )
        
        # 处理非流式返回
        else:
            return await _create_audio_file(
                audio_generator, 
                request.response_format, 
                background_tasks
            )
    
    @router.post("/v1/audio/voices")
    async def create_voice(
        name: str = Form(...),
        reference_text: str = Form(...),
        language: str = Form("zh"),
        description: str = Form(""),
        audio_file: UploadFile = File(...)
    ):
        """创建新音色"""
        
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="必须上传音频文件")
        
        try:
            audio_content = await audio_file.read()
            result = await tts_service.create_voice_from_upload(
                name=name,
                audio_content=audio_content,
                reference_text=reference_text,
                language=language
            )
            return result
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/v1/audio/voice_consents")
    def list_voice_consents(
        limit: int = Query(20, ge=1, le=100),
        after: Optional[str] = Query(None)
    ):
        """列出音色"""
        return tts_service.voice_service.list_voices(limit=limit, after=after)
    
    @router.delete("/v1/audio/voices/{voice_id}")
    def delete_voice(voice_id: str):
        """删除音色"""
        success = tts_service.voice_service.delete_voice(voice_id)
        if not success:
            raise HTTPException(status_code=404, detail="音色不存在")
        return {"message": "删除成功"}
    
    return router

# 辅助函数
async def _openai_sse_formatter(audio_generator):
    """OpenAI SSE 格式化器"""
    async for chunk in audio_generator:
        if "error" in chunk:
            yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
            return
        
        if "audio_data" in chunk:
            # 转换为 int16 并 base64 编码
            audio_int16 = (np.clip(chunk["audio_data"], -1.0, 1.0) * 32767).astype(np.int16)
            b64_audio = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
            
            event_data = {
                "type": "speech.audio.delta",
                "audio": b64_audio
            }
            yield f"data: {json.dumps(event_data)}\n\n"
    
    # 结束事件
    done_data = {
        "type": "speech.audio.done",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }
    yield f"data: {json.dumps(done_data)}\n\n"

async def _binary_audio_stream(audio_generator):
    """二进制音频流"""
    async for chunk in audio_generator:
        if "error" in chunk:
            return
        
        if "audio_data" in chunk:
            # 转换为 int16 PCM
            audio_int16 = (np.clip(chunk["audio_data"], -1.0, 1.0) * 32767).astype(np.int16)
            yield audio_int16.tobytes()

async def _create_audio_file(audio_generator, response_format: str, background_tasks):
    """创建音频文件"""
    task_id = str(uuid.uuid4())
    base_path = f"./tmp/{task_id}"
    os.makedirs("./tmp", exist_ok=True)
    
    # 收集所有音频数据
    all_audio = []
    sample_rate = 24000
    
    async for chunk in audio_generator:
        if "error" in chunk:
            raise HTTPException(status_code=500, detail=chunk["error"])
        
        if "audio_data" in chunk:
            all_audio.extend(chunk["audio_data"])
            sample_rate = chunk.get("sample_rate", 24000)
    
    if not all_audio:
        raise HTTPException(status_code=500, detail="音频生成失败")
    
    # 转换为 numpy 数组
    audio_np = np.array(all_audio, dtype=np.float32)
    
    # 根据格式保存文件
    if response_format == "wav":
        output_file = f"{base_path}.wav"
        sf.write(output_file, audio_np, sample_rate)
        media_type = "audio/wav"
    
    elif response_format == "pcm":
        output_file = f"{base_path}.pcm"
        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
        with open(output_file, "wb") as f:
            f.write(audio_int16.tobytes())
        media_type = "audio/pcm"
    
    else:
        # 使用 ffmpeg 转换其他格式
        wav_file = f"{base_path}.wav"
        output_file = f"{base_path}.{response_format}"
        
        sf.write(wav_file, audio_np, sample_rate)
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_file, output_file
            ], check=True, capture_output=True)
            media_type = f"audio/{response_format}"
        except subprocess.CalledProcessError:
            output_file = wav_file
            media_type = "audio/wav"
    
    # 注册清理任务
    background_tasks.add_task(_cleanup_files, [f"{base_path}.wav", f"{base_path}.pcm", output_file])
    
    return FileResponse(output_file, media_type=media_type)

def _cleanup_files(file_paths: List[str]):
    """清理临时文件"""
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass