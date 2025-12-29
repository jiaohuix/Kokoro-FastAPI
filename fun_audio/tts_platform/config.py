"""
配置管理
"""
import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    type: str
    model_dir: str
    load_trt: bool = False
    fp16: bool = False
    target_sr: int = 24000
    max_val: float = 0.8

@dataclass
class StorageConfig:
    tenants_root: str
    tenant_id: str = "tenant_system"
    temp_dir: str = "./tmp"

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9103
    workers: int = 1

@dataclass
class APIConfig:
    max_text_length: int = 4096
    max_audio_duration: int = 15
    supported_formats: list = None

@dataclass
class Config:
    model: ModelConfig
    storage: StorageConfig
    server: ServerConfig
    api: APIConfig

def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置文件"""
    
    # 默认配置
    default_config = {
        "model": {
            "type": "cosyvoice",
            "cosyvoice": {
                "model_dir": "/path/to/model",
                "load_trt": False,
                "fp16": False,
                "target_sr": 24000,
                "max_val": 0.8
            }
        },
        "storage": {
            "tenants_root": "./audio_data/tenants/",
            "tenant_id": "tenant_system",
            "temp_dir": "./tmp"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 9103,
            "workers": 1
        },
        "api": {
            "max_text_length": 4096,
            "max_audio_duration": 15,
            "supported_formats": ["wav", "mp3", "pcm", "flac", "aac", "opus"]
        }
    }
    
    # 加载配置文件
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
        
        # 合并配置
        config_dict = _merge_config(default_config, file_config)
    else:
        config_dict = default_config
    
    # 处理模型特定配置
    model_type = config_dict["model"]["type"]
    model_specific_config = config_dict["model"].get(model_type, {})
    
    return Config(
        model=ModelConfig(
            type=model_type,
            **model_specific_config
        ),
        storage=StorageConfig(**config_dict["storage"]),
        server=ServerConfig(**config_dict["server"]),
        api=APIConfig(**config_dict["api"])
    )

def _merge_config(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置"""
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    
    return result