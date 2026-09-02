import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def get_ffmpeg_executable(custom_path: Optional[str] = None) -> str:
    """
    Obtiene la ruta del ejecutable de FFmpeg.
    Busca en el orden:
    1. Ruta personalizada (si se proporciona)
    2. Variable de entorno FFMPEG_PATH
    3. FFmpeg instalado en el sistema (PATH)
    4. Binario embebido de imageio-ffmpeg
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_ffmpeg = os.getenv("FFMPEG_PATH")
    if env_ffmpeg and os.path.isfile(env_ffmpeg):
        return env_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    raise RuntimeError(
        "No se encontró el ejecutable de FFmpeg. "
        "Asegúrate de instalar 'imageio-ffmpeg' (pip install imageio-ffmpeg) o tener FFmpeg en el PATH."
    )


def extraer_audio(
    video_path: Union[str, Path],
    output_wav_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_exe: Optional[str] = None,
) -> str:
    """
    Extrae el audio de un video (WebM, MP4, etc.) y lo guarda en formato WAV (PCM 16-bit).
    
    :param video_path: Ruta del archivo de video de entrada.
    :param output_wav_path: Ruta donde se guardará el archivo .wav.
    :param sample_rate: Tasa de muestreo en Hz (por defecto 16000, óptimo para NVIDIA Riva / ASR).
    :param channels: Número de canales (1 para mono, 2 para estéreo).
    :param ffmpeg_exe: Ruta al ejecutable de FFmpeg (opcional).
    :return: Ruta absoluta del archivo WAV generado.
    """
    ffmpeg_bin = get_ffmpeg_executable(ffmpeg_exe)
    video_path = Path(video_path).resolve()
    output_wav_path = Path(output_wav_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",                   # Sobrescribir sin preguntar
        "-i", str(video_path),  # Archivo de entrada (WebM/MP4)
        "-vn",                  # Descartar video
        "-acodec", "pcm_s16le", # Codec WAV PCM de 16 bits
        "-ar", str(sample_rate),# Frecuencia de muestreo
        "-ac", str(channels),   # Canales de audio
        str(output_wav_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error al extraer audio con FFmpeg:\n{result.stderr}"
        )

    return str(output_wav_path)


def extraer_fotogramas(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    fps: float = 0.5,
    quality: int = 2,
    ffmpeg_exe: Optional[str] = None,
) -> List[str]:
    """
    Extrae fotogramas de un video a una tasa de cuadros específica (por defecto 0.5 fps = 1 foto cada 2 segundos).
    
    :param video_path: Ruta del archivo de video de entrada.
    :param output_dir: Carpeta donde se guardarán las imágenes JPG.
    :param fps: Tasa de cuadros por segundo (ej. 0.5 = 1 fotograma cada 2 segundos).
    :param quality: Calidad de compresión JPG (1-31, donde 2-5 es muy alta calidad).
    :param ffmpeg_exe: Ruta al ejecutable de FFmpeg (opcional).
    :return: Lista de rutas absolutas de las imágenes JPG generadas.
    """
    ffmpeg_bin = get_ffmpeg_executable(ffmpeg_exe)
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"El archivo de video no existe: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%04d.jpg"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", str(quality),
        str(pattern)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error al extraer fotogramas con FFmpeg:\n{result.stderr}"
        )

    # Obtener lista ordenada de imágenes generadas
    jpg_files = sorted(str(p) for p in output_dir.glob("frame_*.jpg"))
    return jpg_files


def separar_audio_video(
    video_path: Union[str, Path],
    output_audio_path: Optional[Union[str, Path]] = None,
    output_frames_dir: Optional[Union[str, Path]] = None,
    fps: float = 0.5,
    sample_rate: int = 16000,
    channels: int = 1,
    quality: int = 2,
    ffmpeg_exe: Optional[str] = None,
) -> Dict[str, Union[str, int, List[str]]]:
    """
    Función principal para procesar grabaciones de MediaRecorder (WebM / MP4):
    1. Extrae el audio a un archivo WAV (PCM 16-bit 16kHz mono por defecto).
    2. Extrae fotogramas a una carpeta en formato JPG a intervalos definidos por FPS (por defecto 0.5 fps).
    
    :param video_path: Ruta al archivo de video de entrada (.webm, .mp4, etc.).
    :param output_audio_path: Ruta de salida para el archivo .wav (opcional, por defecto al lado del video).
    :param output_frames_dir: Carpeta de salida para los .jpg (opcional, por defecto al lado del video).
    :param fps: Fotogramas por segundo (0.5 fps = 1 imagen cada 2 segundos).
    :param sample_rate: Frecuencia de muestreo de audio en Hz (por defecto 16000 para ASR/Riva).
    :param channels: Canales de audio (1 para mono, 2 para estéreo).
    :param quality: Calidad de compresión JPG (por defecto 2 = alta calidad).
    :param ffmpeg_exe: Ruta personalizada a FFmpeg si no está en PATH ni en imageio-ffmpeg.
    :return: Diccionario con la información del procesamiento y las rutas generadas.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de video en: {video_path}")

    # Definir rutas por defecto si no fueron provistas
    if output_audio_path is None:
        output_audio_path = video_path.with_suffix(".wav")
    else:
        output_audio_path = Path(output_audio_path).resolve()

    if output_frames_dir is None:
        output_frames_dir = video_path.parent / f"{video_path.stem}_frames"
    else:
        output_frames_dir = Path(output_frames_dir).resolve()

    print(f"--> Procesando video: {video_path.name}")
    print(f"    - Extrayendo audio a: {output_audio_path.name} (Sample rate: {sample_rate}Hz, Canales: {channels})")
    audio_file = extraer_audio(
        video_path=video_path,
        output_wav_path=output_audio_path,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_exe=ffmpeg_exe,
    )

    print(f"    - Extrayendo fotogramas a: {output_frames_dir.name} (Intervalo: {fps} fps)")
    frame_files = extraer_fotogramas(
        video_path=video_path,
        output_dir=output_frames_dir,
        fps=fps,
        quality=quality,
        ffmpeg_exe=ffmpeg_exe,
    )

    print(f"[OK] Proceso completado exitosamente: {len(frame_files)} fotogramas generados.\n")

    return {
        "status": "success",
        "video_path": str(video_path),
        "audio_path": audio_file,
        "frames_dir": str(output_frames_dir),
        "total_frames": len(frame_files),
        "frame_files": frame_files,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_del_video> [fps]")
        print("Ejemplo: python main.py grabacion.webm 0.5")
        sys.exit(1)

    input_file = sys.argv[1]
    input_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    try:
        resultado = separar_audio_video(video_path=input_file, fps=input_fps)
        print("Resultado:")
        for k, v in resultado.items():
            if k != "frame_files":
                print(f"  {k}: {v}")
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
