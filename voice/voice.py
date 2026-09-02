import os
import sys
import string
import wave
import json
from pathlib import Path
from typing import Optional
import numpy as np
import riva.client
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

# Configuration defaults desde variables de entorno
SERVER = os.getenv("NVIDIA_RIVA_SERVER", "grpc.nvcf.nvidia.com:443")
FUNCTION_ID = os.getenv("NVIDIA_FUNCTION_ID", "71203149-d3b7-4460-8231-1be2543a1fca")
API_KEY = os.getenv("NVIDIA_API_KEY")
LANGUAGE_CODE = os.getenv("NVIDIA_LANGUAGE_CODE", "es-US")
DEFAULT_INPUT_FILE = "audio.wav"

# Lista base de muletillas frecuentes en español de crisis
MULETILLAS_TARGET = {"eh", "em", "este", "o sea", "bueno", "digamos", "verdad", "ya"}


def calcular_fonacion_acustica(audio_path: str, frame_duration_ms: int = 30) -> tuple[float, float]:
    """
    Calcula la duración total y el tiempo real de fonación (VAD basado en energía RMS)
    directamente sobre el archivo WAV, sin depender de los timestamps del ASR.
    """
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        total_duration = n_frames / float(sample_rate)
        
        # Leer audio en mono PCM 16-bit
        audio_data = wf.readframes(n_frames)
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    if len(samples) == 0:
        return 0.0, 0.0

    # Tamaño de ventana para análisis (ej. 30 ms)
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    n_chunks = len(samples) // frame_size
    
    # Calcular energía RMS por ventana
    speech_frames = 0
    # Umbral adaptativo: 10% de la energía máxima promedio o piso mínimo
    rms_values = [
        np.sqrt(np.mean(samples[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_chunks)
    ]
    max_rms = np.max(rms_values) if rms_values else 1.0
    threshold = max(300.0, max_rms * 0.08)  # Filtra ruido de fondo

    for rms in rms_values:
        if rms > threshold:
            speech_frames += 1

    tiempo_fonacion_real = speech_frames * (frame_duration_ms / 1000.0)
    return round(total_duration, 2), round(tiempo_fonacion_real, 2)


def evaluar_fluidez_y_diccion(parakeet_output: dict, audio_path: str) -> dict:
    words = parakeet_output.get("words", [])
    total_words = len(words)
    
    # 1. Obtener duración y fonación acústica real
    total_duration, speech_time_real = calcular_fonacion_acustica(audio_path)
    
    if total_words == 0 or total_duration <= 0:
        return {"error": "Audio insuficiente para análisis."}

    # 2. Análisis temporal y pausas corregidas
    pausas = []
    for i in range(total_words - 1):
        # max(0, ...) evita números negativos por solapamientos de ASR
        gap = max(0.0, words[i+1]["start"] - words[i]["end"])
        if gap >= 0.25:  # Considerar pausa si supera 250 ms
            pausas.append(gap)
            
    pausas_vacilacion = [p for p in pausas if p >= 0.70]  # Vacilaciones: >= 700 ms
    pausas_naturales = [p for p in pausas if 0.25 <= p < 0.70]

    # 3. Métricas principales
    wpm = (total_words / total_duration) * 60.0
    wpm_articulacion = (total_words / max(0.1, speech_time_real)) * 60.0
    phonation_ratio = min(1.0, speech_time_real / total_duration)
    avg_confidence = sum(w.get("confidence", 1.0) for w in words) / total_words

    # 4. Detección de muletillas
    muletillas_encontradas = [w["word"].lower() for w in words if w["word"].lower() in MULETILLAS_TARGET]
    densidad_muletillas = (len(muletillas_encontradas) / total_words) * 100.0

    # 5. Rúbrica de Fluidez (Penaliza WPM fuera de rango [120-150], vacilaciones y muletillas)
    penalizacion_wpm = abs(135.0 - wpm) * 0.6
    penalizacion_vacilacion = len(pausas_vacilacion) * 4.0
    penalizacion_muletillas = densidad_muletillas * 5.0
    
    score_fluidez = max(0.0, min(100.0, 100.0 - penalizacion_wpm - penalizacion_vacilacion - penalizacion_muletillas))
    score_diccion = max(0.0, min(100.0, avg_confidence * 100.0))

    return {
        "palabras_totales": total_words,
        "duracion_segundos": total_duration,
        "tiempo_voz_activa_segundos": speech_time_real,
        "wpm_global": round(wpm, 1),
        "wpm_articulacion_neta": round(wpm_articulacion, 1),
        "ratio_fonacion": round(phonation_ratio, 2),
        "pausas_naturales": len(pausas_naturales),
        "pausas_vacilacion": len(pausas_vacilacion),
        "densidad_muletillas_pct": round(densidad_muletillas, 2),
        "muletillas_detectadas": muletillas_encontradas[:10],
        "score_fluidez": round(score_fluidez, 1),
        "score_diccion": round(score_diccion, 1)
    }


def transcribe_and_evaluate(
    audio_path: str = DEFAULT_INPUT_FILE,
    language_code: Optional[str] = None,
    api_key: Optional[str] = None,
    server: Optional[str] = None,
    function_id: Optional[str] = None
):
    key = api_key or os.getenv("NVIDIA_API_KEY") or API_KEY
    if not key:
        print(
            "Error: No se encontró NVIDIA_API_KEY. Define la variable de entorno o agrégala en el archivo .env",
            file=sys.stderr
        )
        return None

    srv = server or os.getenv("NVIDIA_RIVA_SERVER") or SERVER
    func_id = function_id or os.getenv("NVIDIA_FUNCTION_ID") or FUNCTION_ID
    lang_code = language_code or os.getenv("NVIDIA_LANGUAGE_CODE") or LANGUAGE_CODE

    if not os.path.exists(audio_path):
        print(f"Error: No se encontró '{audio_path}'", file=sys.stderr)
        return None

    auth = riva.client.Auth(
        use_ssl=True,
        uri=srv,
        metadata_args=[
            ["function-id", func_id],
            ["authorization", f"Bearer {key}"]
        ]
    )
    asr_service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code=lang_code,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
        ),
        interim_results=False,
    )

    words = []
    full_text = []

    try:
        with riva.client.AudioChunkFileIterator(audio_path, 1600) as audio_chunk_iterator:
            responses = asr_service.streaming_response_generator(
                audio_chunks=audio_chunk_iterator,
                streaming_config=config,
            )
            for response in responses:
                for result in response.results:
                    if not result.is_final:
                        continue
                    for alt in result.alternatives:
                        full_text.append(alt.transcript)
                        for w in alt.words:
                            cleaned_word = w.word.strip(string.punctuation)
                            if cleaned_word:
                                words.append({
                                    "word": cleaned_word,
                                    "start": w.start_time / 1000.0,
                                    "end": w.end_time / 1000.0,
                                    "confidence": w.confidence if w.confidence > 0 else (alt.confidence or 1.0)
                                })

        transcript = " ".join(full_text)
        parakeet_output = {"transcript": transcript, "words": words}

        # Evaluación con la corrección acústica integrada
        metricas = evaluar_fluidez_y_diccion(parakeet_output, audio_path)

        return {
            "transcription": transcript,
            "metrics": metricas
        }

    except Exception as e:
        print(f"Error durante la transcripción: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    res = transcribe_and_evaluate(target_file)
    if res:
        print(json.dumps(res["metrics"], indent=2, ensure_ascii=False))