"""
TTS 模型抽象基类
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import torch
import numpy as np

class TTSModelBase(ABC):
    """TTS 模型抽象基类"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.target_sr = model_config.get("target_sr", 24000)
        
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def inference_zero_shot(
        self, 
        text: str, 
        ref_text: str, 
        ref_audio_path: str,
        speed: float = 1.0,
        stream: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """零样本推理"""
        pass
    
    @abstractmethod
    def inference_instruct(
        self,
        text: str,
        instruct_text: str,
        ref_audio_path: str,
        speed: float = 1.0,
        stream: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """指令推理"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> list:
        """获取支持的音频格式"""
        pass
    
    def preprocess_audio(self, audio_path: str) -> bool:
        """音频预处理，子类可重写"""
        return True
    
    def postprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """音频后处理，子类可重写"""
        return audio_data