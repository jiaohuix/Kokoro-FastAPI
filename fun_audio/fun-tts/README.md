# CosyVoice TTS Docker Deployment

This directory contains Docker deployment files for the CosyVoice TTS service with OpenAI-compatible API.

## Quick Start

1. **Copy environment file and configure paths:**
   ```bash
   cp .env.example .env
   # Edit .env to match your system paths
   ```

2. **Build and start the service:**
   ```bash
   docker-compose up -d --build
   ```

3. **Check service status:**
   ```bash
   docker-compose ps
   docker-compose logs -f cosyvoice-tts
   ```

## Configuration

### Environment Variables

Edit `.env` file to configure:

- `HOST_MODEL_PATH`: Path to your pretrained models directory
- `HOST_AUDIO_DATA_PATH`: Path to your audio data directory  
- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: 0)
- `PORT`: Service port (default: 9102)

### Directory Structure

Ensure your host directories have this structure:

```
${HOST_MODEL_PATH}/
└── CosyVoice3-0.5B-2512-modelscope/  # Model files

${HOST_AUDIO_DATA_PATH}/
└── tenants/
    └── tenant_system/
        └── default/
            └── voices/
                └── [voice_id]/
                    ├── voice.json
                    └── samples/
                        └── [sample_id]/
                            ├── sample.json
                            └── audio.wav
```

## API Usage

### List Available Voices
```bash
curl http://localhost:9102/v1/audio/voice_consents
```

### Generate Speech (OpenAI Compatible)
```bash
curl -X POST http://localhost:9102/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice",
    "input": "Hello, this is a test.",
    "voice": "your_voice_id",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

### Generate Speech (Legacy Format)
```bash
curl -X POST http://localhost:9102/tts \
  -F "tts_text=Hello world" \
  -F "mode=zero_shot" \
  -F "prompt_text=Reference text" \
  -F "prompt_wav=@reference.wav"
```

## Development

For development, you can mount your local code:

```yaml
volumes:
  - ./app.py:/workspace/CosyVoice/app.py
```

## Troubleshooting

1. **Check container logs:**
   ```bash
   docker-compose logs cosyvoice-tts
   ```

2. **Verify GPU access:**
   ```bash
   docker-compose exec cosyvoice-tts nvidia-smi
   ```

3. **Check model loading:**
   ```bash
   docker-compose exec cosyvoice-tts ls -la /workspace/models/
   ```

4. **Test health endpoint:**
   ```bash
   curl http://localhost:9102/v1/audio/voice_consents
   ```

## Stopping the Service

```bash
docker-compose down
```

To also remove the built image:
```bash
docker-compose down --rmi local
```

TODO:

```
python - << 'EOF'
from modelscope import snapshot_download
snapshot_download(
    model_id="pengzhendong/wetext",
    cache_dir="./modelscope_cache"
)
EOF

```