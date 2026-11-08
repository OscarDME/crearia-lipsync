# Servidor de lipsync para Runpod Serverless. CUDA 11.8: corre en hosts con drivers viejos y nuevos.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir "numpy<2" diffusers==0.30.2 accelerate==0.28.0 transformers==4.39.2 "huggingface_hub>=0.19.3,<1.0" \
    librosa==0.11.0 soundfile==0.12.1 einops==0.8.1 opencv-python-headless==4.9.0.80 omegaconf requests runpod
WORKDIR /app
RUN git clone https://github.com/TMElyralab/MuseTalk.git && cd MuseTalk && git checkout 0a89dec45a0192b824e3cf4daf96c239440c5ed8 && rm -rf .git assets
COPY descargar_pesos.py /app/MuseTalk/
RUN cd /app/MuseTalk && python descargar_pesos.py
COPY handler.py /app/
CMD ["python", "-u", "/app/handler.py"]
