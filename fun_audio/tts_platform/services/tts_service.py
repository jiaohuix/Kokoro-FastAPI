"""
TTS 服务层
"""
import os
import uuid
import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ..models.base import TTSModelBase
from .voice_service import VoiceService

class TTSService:
    """TTS 服务"""
    
    def __init__(self, model: TTSModelBase, voice_service: VoiceService):
        self.model = model
        self.voice_service = voice_service
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        stream: bool = False
    ) -> AsyncIterator[Dict[str, Any]]:
        """合成语音"""
        
        # 获取音色参考
        ref_info = self.voice_service.get_voice_reference(voice_id)
        if not ref_info:
            yield {"error": f"未找到音色 {voice_id}"}
            return
        
        # 预处理参考音频
        if not self.model.preprocess_audio(ref_info["ref_audio_path"]):
            yield {"error": "参考音频预处理失败"}
            return
        
        # 选择推理模式
        mode = "instruct" if instructions else "zero_shot"
        
        try:
            # 在线程池中执行同步推理
            loop = asyncio.get_event_loop()
            
            def run_inference():
                if mode == "zero_shot":
                    return self.model.inference_zero_shot(
                        text=text,
                        ref_text=ref_info["ref_text"],
                        ref_audio_path=ref_info["ref_audio_path"],
                        speed=speed,
                        stream=stream
                    )
                else:
                    return self.model.inference_instruct(
                        text=text,
                        instruct_text=instructions,
                        ref_audio_path=ref_info["ref_audio_path"],
                        speed=speed,
                        stream=stream
                    )
            
            model_output = await loop.run_in_executor(self.executor, run_inference)
            
            # 异步迭代结果
            for result in model_output:
                if "error" in result:
                    yield result
                    return
                
                if "audio_data" in result:
                    # 标准化音频数据
                    audio_data = self._normalize_audio(result["audio_data"])
                    yield {
                        "audio_data": audio_data,
                        "sample_rate": result.get("sample_rate", self.model.target_sr)
                    }
                
                # 给事件循环喘息机会
                await asyncio.sleep(0)
                
        except Exception as e:
            logging.error(f"TTS synthesis error: {e}")
            yield {"error": f"合成失败: {str(e)}"}
    
    def _normalize_audio(self, audio_data: Any) -> np.ndarray:
        """标准化音频数据"""
        if isinstance(audio_data, np.ndarray):
            return audio_data.flatten()
        elif hasattr(audio_data, 'numpy'):  # torch.Tensor
            return audio_data.detach().cpu().numpy().flatten()
        elif isinstance(audio_data, (list, tuple)):
            return np.array(audio_data, dtype=np.float32).flatten()
        else:
            return np.array([float(audio_data)], dtype=np.float32)
    
    async def create_voice_from_upload(
        self,
        name: str,
        audio_content: bytes,
        reference_text: str,
        language: str = "zh"
    ) -> Dict[str, str]:
        """从上传的音频创建音色"""
        
        # 保存临时文件
        temp_id = str(uuid.uuid4())
        temp_path = f"/tmp/voice_upload_{temp_id}.wav"
        
        try:
            with open(temp_path, "wb") as f:
                f.write(audio_content)
            
            # 预处理音频
            if not self.model.preprocess_audio(temp_path):
                raise ValueError("音频预处理失败")
            
            # 创建音色
            result = self.voice_service.create_voice(
                name=name,
                audio_file_path=temp_path,
                reference_text=reference_text,
                language=language
            )
            
            return result
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)