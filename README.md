# crearia-lipsync

Servidor de lipsync (MuseTalk 1.5 en modo avatar) para Runpod Serverless.
Cada push a `main` construye la imagen `ghcr.io/oscardme/crearia-lipsync`.

Entrada del trabajo: `avatar`, `avatar_base`, `audio_url`, `upload_url` (URL firmada), `inicio` (opcional).
Los modelos se cargan una vez por máquina, el avatar se restaura una vez y los cuadros van directo a ffmpeg.

MuseTalk: https://github.com/TMElyralab/MuseTalk (MIT; revisar las licencias de sus modelos).
