"""
音色管理服务
"""
import os
import json
import uuid
import shutil
from typing import Optional, Dict, List
from datetime import datetime
import torchaudio

class VoiceService:
    """音色管理服务"""
    
    def __init__(self, tenants_root: str, tenant_id: str = "tenant_system"):
        self.tenants_root = tenants_root
        self.tenant_id = tenant_id
        self.voices_path = os.path.join(tenants_root, tenant_id, "default", "voices")
        os.makedirs(self.voices_path, exist_ok=True)
    
    def create_voice(
        self, 
        name: str, 
        audio_file_path: str, 
        reference_text: str,
        language: str = "zh",
        description: str = ""
    ) -> Dict[str, str]:
        """创建新音色"""
        voice_id = str(uuid.uuid4())
        voice_dir = os.path.join(self.voices_path, voice_id)
        samples_dir = os.path.join(voice_dir, "samples")
        
        # 创建目录结构
        os.makedirs(samples_dir, exist_ok=True)
        
        # 创建样本
        sample_id = str(uuid.uuid4())
        sample_dir = os.path.join(samples_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 复制音频文件
        audio_dest = os.path.join(sample_dir, "audio.wav")
        shutil.copy2(audio_file_path, audio_dest)
        
        # 验证音频文件
        try:
            info = torchaudio.info(audio_dest)
            duration = info.num_frames / info.sample_rate
            if duration > 15:
                raise ValueError("音频时长不能超过15秒")
        except Exception as e:
            shutil.rmtree(voice_dir)
            raise ValueError(f"音频文件无效: {str(e)}")
        
        # 创建样本元数据
        sample_data = {
            "sample_id": sample_id,
            "text": reference_text,
            "is_default_reference": True,
            "created_at": datetime.now().isoformat() + "Z"
        }
        
        with open(os.path.join(sample_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        # 创建音色元数据
        voice_data = {
            "voice_id": voice_id,
            "display_name": name,
            "description": description,
            "language": language,
            "created_at": datetime.now().isoformat() + "Z",
            "status": "active"
        }
        
        with open(os.path.join(voice_dir, "voice.json"), "w", encoding="utf-8") as f:
            json.dump(voice_data, f, ensure_ascii=False, indent=2)
        
        return {
            "voice_id": voice_id,
            "name": name,
            "status": "created"
        }
    
    def get_voice(self, voice_id: str) -> Optional[Dict]:
        """获取音色信息"""
        voice_dir = os.path.join(self.voices_path, voice_id)
        if not os.path.isdir(voice_dir):
            return None
        
        voice_json = os.path.join(voice_dir, "voice.json")
        if not os.path.isfile(voice_json):
            return None
        
        with open(voice_json, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_voice_reference(self, voice_id: str) -> Optional[Dict[str, str]]:
        """获取音色的参考音频和文本"""
        voice_dir = os.path.join(self.voices_path, voice_id)
        if not os.path.isdir(voice_dir):
            return None
        
        samples_dir = os.path.join(voice_dir, "samples")
        if not os.path.isdir(samples_dir):
            return None
        
        # 查找默认参考样本
        chosen = None
        for sample_id in sorted(os.listdir(samples_dir)):
            sample_json = os.path.join(samples_dir, sample_id, "sample.json")
            if not os.path.isfile(sample_json):
                continue
            
            with open(sample_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if data.get("is_default_reference"):
                chosen = data
                break
            if chosen is None:
                chosen = data
        
        if not chosen:
            return None
        
        audio_path = os.path.join(samples_dir, chosen["sample_id"], "audio.wav")
        if not os.path.isfile(audio_path):
            return None
        
        return {
            "ref_text": chosen["text"],
            "ref_audio_path": audio_path
        }
    
    def list_voices(self, limit: int = 20, after: Optional[str] = None) -> Dict:
        """列出音色"""
        if not os.path.exists(self.voices_path):
            return {
                "object": "list",
                "data": [],
                "first_id": None,
                "last_id": None,
                "has_more": False
            }
        
        voices = []
        for voice_id in sorted(os.listdir(self.voices_path)):
            voice_data = self.get_voice(voice_id)
            if voice_data:
                voices.append({
                    "object": "audio.voice_consent",
                    "id": voice_data.get("voice_id", voice_id),
                    "name": voice_data.get("display_name", voice_id),
                    "language": voice_data.get("language"),
                    "created_at": int(datetime.fromisoformat(
                        voice_data.get("created_at", "").replace("Z", "")
                    ).timestamp()) if voice_data.get("created_at") else int(datetime.now().timestamp())
                })
        
        # 分页处理
        start = 0
        if after:
            for i, voice in enumerate(voices):
                if voice["id"] == after:
                    start = i + 1
                    break
        
        page = voices[start:start + limit]
        has_more = start + limit < len(voices)
        
        return {
            "object": "list",
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": has_more
        }
    
    def delete_voice(self, voice_id: str) -> bool:
        """删除音色"""
        voice_dir = os.path.join(self.voices_path, voice_id)
        if os.path.isdir(voice_dir):
            shutil.rmtree(voice_dir)
            return True
        return False