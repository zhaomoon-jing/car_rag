import whisper
import numpy as np
from config import ASR_MODEL_SIZE

model = whisper.load_model(ASR_MODEL_SIZE)

def audio2text(audio_np, sr):
    # gradio传入音频为(采样率, ndarray)
    audio = audio_np.astype(np.float32)
    res = model.transcribe(audio, language="zh")
    return res["text"]