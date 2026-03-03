
# Kokoro 简易部署教程

## 1. 克隆项目

```bash
git clone https://github.com/jiaohuix/Kokoro-FastAPI
cd Kokoro-FastAPI && git checkout dev
```

## 2. 下载模型

**方法一：用 ModelScope 自动下载**
```bash
pip install modelscope
python docker/scripts/download_model_modelscope.py
```

**方法二：手动下载**
```bash
wget https://www.modelscope.cn/models/jiaohui/kokoro/resolve/master/kokoro-v1_0.pth
cp kokoro-v1_0.pth ./api/src/models/v1_0
```

## 3. 拉取 Docker 镜像

**CPU 版：**
```bash
docker pull ghcr.io/remsky/kokoro-fastapi-cpu:latest
```
**GPU 版：**

```bash
docker pull ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

## 4. 启动服务

**CPU 版：**
```bash
docker run -p 8880:8880 \
  -v $(pwd)/api:/app/api \
  ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

**GPU 版：**
```bash
docker run --gpus all -p 8880:8880 \
  -v $(pwd)/api:/app/api \
  ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

## 5. 访问和调用

- API 文档: [http://localhost:8880/docs](http://localhost:8880/docs)
- Web UI: [http://localhost:8880/web](http://localhost:8880/web)

### Python 示例（OpenAI 兼容流式 TTS）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

# 流式写入文件
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    input="Hello world!"
) as response:
    response.stream_to_file("output.mp3")

# 实时播放（需要 PyAudio）
import pyaudio
player = pyaudio.PyAudio().open(
    format=pyaudio.paInt16, channels=1, rate=24000, output=True
)
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_bella",
    response_format="pcm",
    input="Hello world!"
) as response:
    for chunk in response.iter_bytes(chunk_size=1024):
        player.write(chunk)
```

## 6. 打包镜像cu12.3

```
cd docker/gpu
docker compose build --no-cache
sudo chown -R 1001:1001 api/src/models/v1_0
docker save kokoro-fastapi:gpu-cu123 | gzip > kokoro-fastapi-gpu-cu123.tgz
```

---

## 常见问题

- **中文断句**：本项目已支持中文标点断句，合成更自然。
- **模型文件**：请确保 `kokoro-v1_0.pth` 已放到 `./api/src/models/v1_0/` 目录下。
- **依赖问题**：如遇缺包报错，进入容器后 `pip install 包名` 即可。


如需详细功能和高级用法，请参考 [原始项目](https://github.com/remsky/Kokoro-FastAPI)。
