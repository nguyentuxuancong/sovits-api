import os
import io
import librosa
import numpy as np
from pydub import AudioSegment
from logging import getLogger
from so_vits_svc_fork.inference.core import Svc
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi import status
from fastapi.responses import Response
import uvicorn


app = FastAPI()

LOG = getLogger(__name__)
SRC_DIR = os.path.dirname(os.path.dirname(__file__))
DEVICE = os.getenv("DEVICE", "cpu")

LOG.info(f"Loading SVC model on {DEVICE}")
svc_model = Svc(
    net_g_path=os.path.join(SRC_DIR, "models", "generator.pth"),
    config_path=os.path.join(SRC_DIR, "models", "config.json"),
    device=DEVICE
)

LOG.info(f"Model loaded successfully on {DEVICE}")


@app.post("/infer/")
def inference(file: UploadFile, speaker: int):
    if file is None or 0 > speaker or speaker > len(svc_model.spk2id):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                             detail=f"Input must be a valid audio file and speaker id must be in range [0, {len(svc_model.spk2id)}]")
    allowed_extensions = {"mp3", "wav", "flac"}
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() not in allowed_extensions:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                             detail="Unsupported file format. Supported formats: .mp3, .wav, .flac")

    try:
        audio, _ = librosa.load(file.file, sr=svc_model.target_sample)
    except Exception:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                             detail="Input file is invalid. Please check and try again.")
    LOG.info(f"Audio file loaded successfully. Length: {len(audio) / svc_model.target_sample}s")
    audio = svc_model.infer_silence(
        audio.astype(np.float32),
        speaker=speaker,
        auto_predict_f0=True,
        db_thresh=-20
    )
    audio_pcm = (audio * 32767).astype('int16')
    audio = AudioSegment(data=audio_pcm.tobytes(), sample_width=2, frame_rate=svc_model.target_sample, channels=1)
    buffer = io.BytesIO()
    audio.export(buffer, format="mp3")

    return Response(content=buffer.getvalue(), media_type="audio/mpeg")
