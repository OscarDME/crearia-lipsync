"""Servidor de lipsync (MuseTalk 1.5, modo avatar) para Runpod Serverless.

Recibe: avatar (id), avatar_base (URL pública de la carpeta de avatares preparados),
audio_url, upload_url (URL firmada para subir el MP4) y opcional inicio (cuadro del ciclo donde empieza).
Devuelve tiempos y datos del video. No guarda llaves: sube con la URL firmada que le da el motor.

Diferencias con scripts/realtime_inference.py (mismo resultado, mucho más rápido):
- los modelos se cargan una vez por máquina, no por video;
- el avatar se baja y restaura una vez por máquina y queda en disco para los siguientes;
- los cuadros no se guardan como PNG: van directo a ffmpeg por una tubería;
- la mezcla de la boca se hace solo en el recuadro de la cara y en varios hilos.
"""
import glob
import hashlib
import os
import pickle
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import runpod
import torch

MT = "/app/MuseTalk"
sys.path.insert(0, MT)
os.chdir(MT)

from transformers import WhisperModel  # noqa: E402
from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.utils import load_all_model  # noqa: E402

FPS = 25
LOTE = int(os.environ.get("LOTE", "32"))
HILOS = int(os.environ.get("HILOS", str(max(2, min(16, (os.cpu_count() or 4))))))
CACHE = os.environ.get("CACHE_AVATARES", "/tmp/avatares")
os.makedirs(CACHE, exist_ok=True)

t_carga = time.time()
device = torch.device("cuda")
vae, unet, pe = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="models/musetalkV15/musetalk.json",
    device=device,
)
timesteps = torch.tensor([0], device=device)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
whisper = WhisperModel.from_pretrained("models/whisper").to(device=device, dtype=torch.float16).eval()
whisper.requires_grad_(False)
# Calentamiento: la primera llamada de librosa/whisper es lenta; se paga una vez al arrancar la máquina.
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", "/tmp/silencio.wav"], check=True)
with torch.no_grad():
    _f, _l = audio_processor.get_audio_feature("/tmp/silencio.wav", weight_dtype=torch.float16)
    audio_processor.get_whisper_chunk(_f, device, torch.float16, whisper, _l, fps=FPS)
SEG_CARGA = round(time.time() - t_carga, 1)
print(f"modelos cargados en {SEG_CARGA}s", flush=True)


def _codificador():
    """h264_nvenc si la tarjeta lo permite (mucho más rápido); si no, libx264."""
    prueba = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        if subprocess.run(prueba, capture_output=True, timeout=30).returncode == 0:
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "24", "-b:v", "5M", "-maxrate", "7M", "-bufsize", "14M"]
    except Exception:
        pass
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-maxrate", "7M", "-bufsize", "14M"]


CODIFICADOR = _codificador()
print("codificador:", CODIFICADOR[1], flush=True)

AVATARES = {}  # en memoria, por id


def _bajar(url, destino):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for parte in r.iter_content(1 << 20):
                f.write(parte)


def _bajar_o_nada(url, destino):
    try:
        _bajar(url, destino)
        return True
    except requests.HTTPError:
        return False


def cargar_avatar(avatar, base, t):
    """Deja el avatar listo en memoria. Anota en t cuánto tardó cada paso."""
    if avatar in AVATARES:
        t["avatar_desde"] = "memoria"
        return AVATARES[avatar]
    AVATARES.clear()  # una sola a la vez en memoria (los cuadros pesan ~8 GB)
    d = os.path.join(CACHE, avatar)
    t["avatar_desde"] = "disco"
    if not os.path.exists(os.path.join(d, "listo")):
        t["avatar_desde"] = "R2"
        os.makedirs(d, exist_ok=True)
        x = time.time()
        # Todo se baja al mismo tiempo; las partes que no existen se ignoran.
        with ThreadPoolExecutor(8) as pool:
            clip = pool.submit(_bajar, f"{base}/clips/{avatar}.mp4", f"{d}/clip.mp4")
            sha = pool.submit(_bajar, f"{base}/preparados/{avatar}.tar.sha256", f"{d}/sha")
            partes = [pool.submit(_bajar_o_nada, f"{base}/preparados/{avatar}.tar.parte-{n:02d}", f"{d}/p{n:02d}") for n in range(6)]
            clip.result(); sha.result()
            hay = [k for k, f in enumerate(partes) if f.result()]
        if not hay or hay != list(range(len(hay))):
            raise RuntimeError(f"partes del avatar incompletas: {hay}")
        t["avatar_bajar_s"] = round(time.time() - x, 1)
        x = time.time()
        h = hashlib.sha256()
        tar = f"{d}/a.tar"
        with open(tar, "wb") as salida:
            for k in hay:
                with open(f"{d}/p{k:02d}", "rb") as f:
                    for b in iter(lambda: f.read(1 << 22), b""):
                        h.update(b)
                        salida.write(b)
                os.remove(f"{d}/p{k:02d}")
        if h.hexdigest() != open(f"{d}/sha").read().split()[0]:
            raise RuntimeError("paquete del avatar corrupto")
        with tarfile.open(tar) as tf:
            tf.extractall(d)
        os.remove(tar)
        open(f"{d}/listo", "w").write("1")
        t["avatar_extraer_s"] = round(time.time() - x, 1)

    a = os.path.join(d, avatar)
    x = time.time()
    # Los cuadros del clip van a memoria con ffmpeg en todos los núcleos (solo la ida; la vuelta se lee al revés).
    sonda = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", f"{d}/clip.mp4"], capture_output=True, text=True, check=True)
    ancho, alto = [int(v) for v in sonda.stdout.strip().split(",")[:2]]
    tam = ancho * alto * 3
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "0", "-i", f"{d}/clip.mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, bufsize=tam * 4)
    cuadros = []
    while True:
        b = dec.stdout.read(tam)
        if len(b) < tam:
            break
        cuadros.append(np.frombuffer(b, np.uint8).reshape(alto, ancho, 3))
    dec.wait()
    if not cuadros:
        raise RuntimeError("no se pudo leer el clip")
    n = len(cuadros)
    t["avatar_decodificar_s"] = round(time.time() - x, 1)
    x = time.time()
    latentes = torch.load(f"{a}/latents.pt", map_location="cpu")
    coords = pickle.load(open(f"{a}/coords.pkl", "rb"))
    caja_mascara = pickle.load(open(f"{a}/mask_coords.pkl", "rb"))
    archivos = sorted(glob.glob(f"{a}/mask/*.png"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    with ThreadPoolExecutor(HILOS) as pool:
        mascaras = list(pool.map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), archivos))
    t["avatar_mascaras_s"] = round(time.time() - x, 1)
    if not (len(coords) == len(caja_mascara) == len(mascaras) == 2 * n):
        raise RuntimeError(f"avatar incompleto: cuadros {n}, coords {len(coords)}, mascaras {len(mascaras)}")
    if len(latentes) != len(coords):
        raise RuntimeError(f"latentes ({len(latentes)}) no coinciden con cuadros ({len(coords)})")
    av = {
        "n": n,
        "cuadros": cuadros,
        "lat": torch.cat(latentes, dim=0).to(device=device, dtype=torch.float16),
        "coords": coords,
        "cajas": caja_mascara,
        "mascaras": mascaras,
        "alto": cuadros[0].shape[0],
        "ancho": cuadros[0].shape[1],
    }
    AVATARES[avatar] = av
    return av


def cuadro_base(av, p):
    n = av["n"]
    return av["cuadros"][p] if p < n else av["cuadros"][2 * n - 1 - p]


def mezclar(av, p, cara):
    """Misma mezcla que get_image_blending de MuseTalk, pero solo en el recuadro y con numpy."""
    base = cuadro_base(av, p)
    out = np.array(base, copy=True)
    x, y, x1, y1 = [int(v) for v in av["coords"][p]]
    if x1 - x <= 0 or y1 - y <= 0:
        return out
    cara = cv2.resize(cara.astype(np.uint8), (x1 - x, y1 - y))
    xs, ys, xe, ye = [int(v) for v in av["cajas"][p]]
    H, W = out.shape[:2]
    grande = np.zeros((ye - ys, xe - xs, 3), np.uint8)
    cx0, cy0, cx1, cy1 = max(xs, 0), max(ys, 0), min(xe, W), min(ye, H)
    grande[cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs] = out[cy0:cy1, cx0:cx1]
    grande[y - ys:y1 - ys, x - xs:x1 - xs] = cara
    a = av["mascaras"][p][cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs, None].astype(np.float32) / 255.0
    zona = out[cy0:cy1, cx0:cx1].astype(np.float32)
    nueva = grande[cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs].astype(np.float32)
    out[cy0:cy1, cx0:cx1] = (nueva * a + zona * (1.0 - a) + 0.5).astype(np.uint8)
    return out


@torch.no_grad()
def handler(job):
    e = job["input"]
    t = {"carga_modelos_s": SEG_CARGA}
    t0 = time.time()
    trabajo = f"/tmp/trabajo-{job['id']}"
    os.makedirs(trabajo, exist_ok=True)
    try:
        av = cargar_avatar(e["avatar"], e["avatar_base"].rstrip("/"), t)
        t["avatar_s"] = round(time.time() - t0, 1)

        t1 = time.time()
        audio = f"{trabajo}/audio"
        _bajar(e["audio_url"], audio)
        t["audio_bajar_s"] = round(time.time() - t1, 1)
        audio16 = f"{trabajo}/audio16.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", audio, "-ac", "1", "-ar", "16000", audio16], check=True)
        feats, largo = audio_processor.get_audio_feature(audio16, weight_dtype=torch.float16)
        trozos = audio_processor.get_whisper_chunk(feats, device, torch.float16, whisper, largo, fps=FPS,
                                                   audio_padding_length_left=2, audio_padding_length_right=2)
        total = len(trozos)
        t["audio_s"] = round(time.time() - t1, 1)

        salida = f"{trabajo}/video.mp4"
        W, H = av["ancho"], av["alto"]
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-i", audio, *CODIFICADOR, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
             "-movflags", "+faststart", salida],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        L = 2 * av["n"]
        inicio = int(e.get("inicio", 0))
        t2 = time.time()
        gpu_s = 0.0
        with ThreadPoolExecutor(HILOS) as pool:
            pendiente = None
            for a in range(0, total, LOTE):
                b = min(total, a + LOTE)
                pos = [(inicio + i) % L for i in range(a, b)]
                tg = time.time()
                lat = av["lat"][pos]
                aud = pe(trozos[a:b].to(device))
                pred = unet.model(lat, timesteps, encoder_hidden_states=aud).sample
                caras = vae.decode_latents(pred)
                gpu_s += time.time() - tg
                # mientras la tarjeta hace el siguiente lote, los hilos mezclan este
                if pendiente is not None:
                    for c in pendiente:
                        ff.stdin.write(c.result().tobytes())
                pendiente = [pool.submit(mezclar, av, p, caras[k]) for k, p in enumerate(pos)]
            if pendiente is not None:
                for c in pendiente:
                    ff.stdin.write(c.result().tobytes())
        ff.stdin.close()
        err = ff.stderr.read().decode(errors="ignore")
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg: {err[-500:]}")
        t["generar_s"] = round(time.time() - t2, 1)
        t["gpu_s"] = round(gpu_s, 1)

        t3 = time.time()
        with open(salida, "rb") as f:
            r = requests.put(e["upload_url"], data=f, headers={"Content-Type": "video/mp4"}, timeout=600)
        r.raise_for_status()
        t["subir_s"] = round(time.time() - t3, 1)
        t["total_s"] = round(time.time() - t0, 1)
        return {
            "cuadros": total,
            "segundos": round(total / FPS, 2),
            "siguiente_inicio": (inicio + total) % L,
            "peso_mb": round(os.path.getsize(salida) / 1e6, 1),
            "codificador": CODIFICADOR[1],
            "hilos": HILOS,
            "nucleos": os.cpu_count(),
            "vram_max_mb": round(torch.cuda.max_memory_allocated() / 1e6),
            "gpu": torch.cuda.get_device_name(0),
            "tiempos": t,
        }
    finally:
        subprocess.run(["rm", "-rf", trabajo])


runpod.serverless.start({"handler": handler})
