# Baja solo los pesos que hacen falta para GENERAR (la preparación de avatares ya está hecha).
from huggingface_hub import hf_hub_download as d
d("TMElyralab/MuseTalk", "musetalkV15/unet.pth", local_dir="models")
d("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", local_dir="models")
for f in ("config.json", "diffusion_pytorch_model.bin"):
    d("stabilityai/sd-vae-ft-mse", f, local_dir="models/sd-vae")
for f in ("config.json", "pytorch_model.bin", "preprocessor_config.json"):
    d("openai/whisper-tiny", f, local_dir="models/whisper")
print("pesos ok")
