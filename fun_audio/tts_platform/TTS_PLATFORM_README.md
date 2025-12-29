# TTS Platform - 解耦架构设计

## 架构概述

这是一个解耦的 TTS 平台，支持多种 TTS 模型和 OpenAI 兼容接口。

### 核心特性

- **模型解耦**: 通过抽象基类支持多种 TTS 模型
- **OpenAI 兼容**: 完全兼容 OpenAI Speech API
- **音色管理**: 支持上传音频文件创建自定义音色
- **流式输出**: 支持实时流式语音合成
- **配置驱动**: 通过 YAML 配置文件管理所有设置

## 目录结构

```
tts_platform/
├── models/                 # 模型层
│   ├── base.py            # TTS 模型抽象基类
│   └── cosyvoice_model.py # CosyVoice 模型实现
├── services/              # 服务层
│   ├── voice_service.py   # 音色管理服务
│   └── tts_service.py     # TTS 合成服务
├── api/                   # API 层
│   └── openai_compatible.py # OpenAI 兼容接口
└── config.py              # 配置管理
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn pydantic soundfile librosa torchaudio
```

### 2. 配置模型

编辑 `config.yaml`:

```yaml
model:
  type: cosyvoice
  cosyvoice:
    model_dir: "/path/to/your/cosyvoice/model"
    load_trt: false
    fp16: false
```

### 3. 启动服务

```bash
python app_refactored.py
```

### 4. 创建音色

```bash
curl -X POST "http://localhost:9103/v1/audio/voices" \
  -F "name=我的音色" \
  -F "reference_text=这是参考文本" \
  -F "language=zh" \
  -F "audio_file=@your_audio.wav"
```

### 5. 语音合成

```bash
curl -X POST "http://localhost:9103/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice",
    "input": "你好，这是测试文本",
    "voice": "your_voice_id",
    "response_format": "wav"
  }' \
  --output output.wav
```

## API 接口

### 音色管理

#### 创建音色
```
POST /v1/audio/voices
```

参数:
- `name`: 音色名称
- `reference_text`: 参考文本
- `language`: 语言 (zh/en)
- `audio_file`: 音频文件 (最长15秒)

#### 列出音色
```
GET /v1/audio/voice_consents?limit=20&after=voice_id
```

#### 删除音色
```
DELETE /v1/audio/voices/{voice_id}
```

### 语音合成

#### OpenAI 兼容接口
```
POST /v1/audio/speech
```

参数:
- `model`: 模型名称 (cosyvoice)
- `input`: 待合成文本 (最长4096字符)
- `voice`: 音色ID
- `instructions`: 指令文本 (可选)
- `response_format`: 输出格式 (wav/mp3/pcm等)
- `speed`: 语速 (0.25-4.0)
- `stream`: 是否流式输出
- `stream_format`: 流式格式 (audio/sse)

## 扩展新模型

### 1. 实现模型类

```python
from tts_platform.models.base import TTSModelBase

class YourModel(TTSModelBase):
    def load_model(self):
        # 加载你的模型
        pass
    
    def inference_zero_shot(self, text, ref_text, ref_audio_path, speed, stream):
        # 实现零样本推理
        for result in your_model_inference():
            yield {"audio_data": result, "sample_rate": self.target_sr}
    
    def inference_instruct(self, text, instruct_text, ref_audio_path, speed, stream):
        # 实现指令推理
        pass
```

### 2. 注册模型

在 `app_refactored.py` 中添加模型选择逻辑:

```python
if config.model.type == "cosyvoice":
    model = CosyVoiceModel(model_config)
elif config.model.type == "your_model":
    model = YourModel(model_config)
```

## 配置选项

### 模型配置
```yaml
model:
  type: cosyvoice  # 模型类型
  cosyvoice:       # 模型特定配置
    model_dir: "/path/to/model"
    load_trt: false
    fp16: false
    target_sr: 24000
    max_val: 0.8
```

### 存储配置
```yaml
storage:
  tenants_root: "./audio_data/tenants/"
  tenant_id: "tenant_system"
  temp_dir: "./tmp"
```

### 服务配置
```yaml
server:
  host: "0.0.0.0"
  port: 9103
  workers: 1
```

## 使用示例

查看 `example_usage.py` 了解完整的使用示例，包括:

- 创建音色
- 列出音色
- 语音合成
- 流式合成

## 与原版对比

### 优势

1. **解耦设计**: 模型、服务、API 分离，易于维护和扩展
2. **标准化接口**: 统一的 TTS 模型抽象，支持多种模型
3. **完整的音色管理**: 支持音色的创建、列表、删除
4. **配置驱动**: 通过配置文件管理所有设置
5. **错误处理**: 完善的异常处理和日志记录
6. **资源管理**: 自动清理临时文件

### 迁移指南

从原版迁移到新架构:

1. 更新配置文件 `config.yaml`
2. 使用 `app_refactored.py` 启动服务
3. 音色管理从文件系统操作改为 API 调用
4. TTS 调用保持 OpenAI 兼容格式不变

## 性能优化

- 使用线程池处理模型推理
- 支持流式输出减少延迟
- 音频预处理优化
- 临时文件自动清理

## 安全考虑

- 文件上传大小限制
- 音频时长限制
- 输入文本长度限制
- 临时文件安全清理