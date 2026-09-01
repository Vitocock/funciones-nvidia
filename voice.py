import os
import sys
import string
import wave
import json
import riva.client

# Configuration defaults
SERVER = "grpc.nvcf.nvidia.com:443"
FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"  # NVIDIA Parakeet Multilingual ASR
API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)
LANGUAGE_CODE = "es-US"
DEFAULT_INPUT_FILE = "audio.wav"

def get_audio_duration(audio_path: str) -> float:
    """Calcula la duración del archivo de audio en segundos."""
    try:
        with wave.open(audio_path, 'rb') as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0

def evaluar_fluidez_y_diccion(parakeet_output: dict) -> dict:
    """
    Evalúa la fluidez y dicción a partir de los timestamps y métricas
    proporcionados por el modelo NVIDIA Parakeet.
    """
    words = parakeet_output.get("words", [])
    total_words = len(words)
    total_duration = parakeet_output.get("audio_duration", 0.0)
    
    if total_words == 0 or total_duration <= 0:
        return {
            "error": "No se encontraron suficientes palabras o duración válida para evaluar."
        }
    
    # 1. Tiempos y pausas
    speech_time = sum(w["end"] - w["start"] for w in words)
    pauses = [words[i+1]["start"] - words[i]["end"] for i in range(total_words - 1)]
    hesitations = [p for p in pauses if p > 1.0]  # Pausas largas (> 1 segundo)
    
    # 2. Métricas
    wpm = (total_words / total_duration) * 60
    phonation_ratio = speech_time / total_duration
    avg_confidence = sum(w.get("confidence", 1.0) for w in words) / total_words
    
    # 3. Mapeo a puntaje (Rúbrica 0 a 100)
    score_fluidez = max(0.0, min(100.0, 100.0 - (len(hesitations) * 8.0) - abs(135.0 - wpm) * 0.8))
    score_diccion = max(0.0, min(100.0, avg_confidence * 100.0))
    
    return {
        "palabras_totales": total_words,
        "duracion_segundos": round(total_duration, 2),
        "wpm": round(wpm, 1),
        "pausas_largas": len(hesitations),
        "ratio_fonacion": round(phonation_ratio, 2),
        "score_fluidez": round(score_fluidez, 1),
        "score_diccion": round(score_diccion, 1)
    }

def transcribe_and_evaluate(audio_path: str = DEFAULT_INPUT_FILE, language_code: str = LANGUAGE_CODE):
    if not os.path.exists(audio_path):
        print(f"Error: No se encontró el archivo de audio '{audio_path}'", file=sys.stderr)
        return None

    print(f"--> Procesando '{audio_path}' (Idioma: {language_code})...")
    
    # Configurar autenticación con NVIDIA Cloud Functions (NVCF)
    auth = riva.client.Auth(
        use_ssl=True,
        uri=SERVER,
        metadata_args=[
            ["function-id", FUNCTION_ID],
            ["authorization", f"Bearer {API_KEY}"]
        ]
    )
    
    asr_service = riva.client.ASRService(auth)
    
    # Habilitar word_time_offsets para obtener los timestamps de cada palabra
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code=language_code,
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
        print("\n=== Transcripción ===")
        print(transcript)
        print("=====================\n")

        # Determinar duración del audio
        duration = get_audio_duration(audio_path)
        if duration <= 0 and words:
            duration = words[-1]["end"]

        parakeet_output = {
            "transcript": transcript,
            "words": words,
            "audio_duration": duration
        }

        # Evaluar fluidez y dicción
        metricas = evaluar_fluidez_y_diccion(parakeet_output)
        
        print("=== Evaluación de Fluidez y Dicción ===")
        print(f" • Palabras detectadas : {metricas.get('palabras_totales')}")
        print(f" • Duración del audio  : {metricas.get('duracion_segundos')} s")
        print(f" • Velocidad (WPM)     : {metricas.get('wpm')} palabras/min")
        print(f" • Pausas largas (>1s) : {metricas.get('pausas_largas')}")
        print(f" • Ratio de fonación   : {metricas.get('ratio_fonacion')}")
        print(f" • Score de Fluidez    : {metricas.get('score_fluidez')} / 100")
        print(f" • Score de Dicción    : {metricas.get('score_diccion')} / 100")
        print("=======================================\n")
        
        return {
            "transcription": transcript,
            "parakeet_output": parakeet_output,
            "metrics": metricas
        }

    except Exception as e:
        print(f"\nError durante la transcripción: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    transcribe_and_evaluate(target_file)
