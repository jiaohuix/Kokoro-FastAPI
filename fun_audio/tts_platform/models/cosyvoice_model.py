"""
CosyVoice 模型实现
"""
import os
import torch
import librosa
import torchaudio
import numpy as np
from typing import Iterator, Dict, Any
from .base import TTSModelBase
from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceAutoModel
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav

class CosyVoiceModel(TTSModelBase):
    """CosyVoice 模型实现"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.max_val = model_config.get("max_val", 0.8)
        
    def load_model(self) -> None:
        """加载 CosyVoice 模型"""
        model_dir = self.model_config["model_dir"]
        load_trt = self.model_config.get("load_trt", False)
        fp16 = self.model_config.get("fp16", False)
        
        self.model = CosyVoiceAutoModel(
            model_dir=model_dir,
            load_trt=load_trt,
            fp16=fp16
        )
    
    def inference_zero_shot(
        self, 
        text: str, 
        ref_text: str, 
        ref_audio_path: str,
        speed: float = 1.0,
        stream: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """零样本推理"""
        set_all_random_seed(0)
        
        # CosyVoice 特定的提示词格式
        full_prompt_text = 'You are a helpful assistant.<|endofprompt|>' + ref_text
        
        try:
            model_output = self.model.inference_zero_shot(
                text, full_prompt_text, ref_audio_path, 
                stream=stream, speed=speed
            )
            
            for res in model_output:
                if 'tts_speech' in res:
                    yield {
                        "audio_data": res['tts_speech'].numpy().flatten(),
                        "sample_rate": self.target_sr
                    }
        except Exception as e:
            yield {"error": f"CosyVoice inference error: {str(e)}"}
    
    def inference_instruct(
        self,
        text: str,
        instruct_text: str,
        ref_audio_path: str,
        speed: float = 1.0,
        stream: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """指令推理"""
        set_all_random_seed(0)
        
        try:
            model_output = self.model.inference_instruct2(
                text, instruct_text, ref_audio_path,
                stream=stream, speed=speed
            )
            
            for res in model_output:
                if 'tts_speech' in res:
                    yield {
                        "audio_data": res['tts_speech'].numpy().flatten(),
                        "sample_rate": self.target_sr
                    }
        except Exception as e:
            yield {"error": f"CosyVoice instruct error: {str(e)}"}
    
    def get_supported_formats(self) -> list:
        """获取支持的音频格式"""
        return ["wav", "mp3", "pcm", "flac"]
    
    def preprocess_audio(self, audio_path: str) -> bool:
        """CosyVoice 特定的音频预处理"""
        try:
            speech = load_wav(audio_path, target_sr=self.target_sr, min_sr=16000)
            
            # trim 处理
            speech_trimmed, _ = librosa.effects.trim(
                speech.numpy(), top_db=60, frame_length=440, hop_length=220
            )
            speech = torch.from_numpy(speech_trimmed)
            
            # 归一化
            if speech.abs().max() > self.max_val:
                speech = speech / speech.abs().max() * self.max_val
            
            # 补静音防止截断
            speech = torch.concat([speech, torch.zeros(1, int(self.target_sr * 0.2))], dim=1)
            torchaudio.save(audio_path, speech, self.target_sr)
            
            return True
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            return False