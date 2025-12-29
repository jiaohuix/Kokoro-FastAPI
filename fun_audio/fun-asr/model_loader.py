# model_loader.py
import os
from dotenv import load_dotenv
from model import FunASRNano

load_dotenv()  # 读取 .env

MODEL_DIR = os.getenv("MODEL_DIR", "FunAudioLLM/Fun-ASR-Nano-2512")
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda:0")

_model = None
_kwargs = None

def get_model():
    global _model, _kwargs
    if _model is None:
        _model, _kwargs = FunASRNano.from_pretrained(
            model=MODEL_DIR,
            device=CUDA_DEVICE
        )
        _model.eval()
    return _model, _kwargs

