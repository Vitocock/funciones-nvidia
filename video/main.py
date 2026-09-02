import os
import sys
import math
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# URLs de los modelos oficiales de MediaPipe
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
POSE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# Constantes de puntos clave (Landmarks)
# Face Mesh / Iris
IRIS_LEFT_CENTER = 468
IRIS_RIGHT_CENTER = 473
EYE_LEFT_OUTER = 33
EYE_LEFT_INNER = 133
EYE_RIGHT_INNER = 362
EYE_RIGHT_OUTER = 263
EYE_LEFT_TOP = 159
EYE_LEFT_BOTTOM = 145
EYE_RIGHT_TOP = 386
EYE_RIGHT_BOTTOM = 374
NOSE_TIP = 1
CHIN = 152

# Pose
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12


def _descargar_modelo_si_no_existe(ruta_modelo: Path, url: str) -> str:
    """Descarga el archivo del modelo MediaPipe si no existe localmente."""
    ruta_modelo = Path(ruta_modelo).resolve()
    if not ruta_modelo.exists():
        ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
        print(f"--> Descargando modelo MediaPipe desde: {url}...")
        urllib.request.urlretrieve(url, str(ruta_modelo))
        print(f"[OK] Modelo guardado en: {ruta_modelo}")
    return str(ruta_modelo)


class EvaluadorExpresionVideo:
    """
    Evaluador de presencia, contacto visual, postura y microexpresiones
    utilizando MediaPipe FaceLandmarker y PoseLandmarker.
    """

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir = Path(models_dir)

        face_model_path = _descargar_modelo_si_no_existe(
            self.models_dir / "face_landmarker.task",
            FACE_LANDMARKER_URL
        )
        pose_model_path = _descargar_modelo_si_no_existe(
            self.models_dir / "pose_landmarker.task",
            POSE_LANDMARKER_URL
        )

        # Inicializar FaceLandmarker con Blendshapes
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # Inicializar PoseLandmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            num_poses=1,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    def _extraer_blendshapes_dict(self, blendshapes_list) -> Dict[str, float]:
        """Convierte la lista de blendshapes de MediaPipe en un diccionario {nombre: score}."""
        if not blendshapes_list or len(blendshapes_list) == 0:
            return {}
        return {cat.category_name: cat.score for cat in blendshapes_list[0]}

    def _evaluar_contacto_visual(
        self,
        face_landmarks,
        blendshapes: Dict[str, float]
    ) -> Tuple[bool, float, float, str]:
        """
        Eje 1: Contacto Visual (Gaze Tracking)
        Calcula el ratio horizontal del iris respecto a las comisuras del ojo (Rx).
        Condición: 0.40 < Rx < 0.60 en ambos ojos y ojos abiertos.
        """
        # Coordenadas X del ojo izquierdo
        x_iris_l = face_landmarks[IRIS_LEFT_CENTER].x
        x_inner_l = face_landmarks[EYE_LEFT_INNER].x
        x_outer_l = face_landmarks[EYE_LEFT_OUTER].x

        min_x_l = min(x_inner_l, x_outer_l)
        max_x_l = max(x_inner_l, x_outer_l)
        span_l = max_x_l - min_x_l
        rx_left = (x_iris_l - min_x_l) / (span_l + 1e-6)

        # Coordenadas X del ojo derecho
        x_iris_r = face_landmarks[IRIS_RIGHT_CENTER].x
        x_inner_r = face_landmarks[EYE_RIGHT_INNER].x
        x_outer_r = face_landmarks[EYE_RIGHT_OUTER].x

        min_x_r = min(x_inner_r, x_outer_r)
        max_x_r = max(x_inner_r, x_outer_r)
        span_r = max_x_r - min_x_r
        rx_right = (x_iris_r - min_x_r) / (span_r + 1e-6)

        # Apertura vertical / Parpadeo
        blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_right = blendshapes.get("eyeBlinkRight", 0.0)
        ojos_abiertos = blink_left < 0.60 and blink_right < 0.60

        # Contacto directo: ambos ojos con iris centrado (0.40 < Rx < 0.60) y abiertos
        mirada_directa = bool(
            ojos_abiertos and (0.40 <= rx_left <= 0.60) and (0.40 <= rx_right <= 0.60)
        )

        detalle = "mirada_directa"
        if not ojos_abiertos:
            detalle = "ojos_cerrados"
        elif rx_left < 0.40 or rx_right < 0.40:
            detalle = "mirada_hacia_lado_izquierdo"
        elif rx_left > 0.60 or rx_right > 0.60:
            detalle = "mirada_hacia_lado_derecho"

        return mirada_directa, rx_left, rx_right, detalle

    def _evaluar_alineacion(
        self,
        face_landmarks,
        pose_landmarks
    ) -> Tuple[float, float, str, str]:
        """
        Eje 2: Postura y Alineación Corporal
        1. Simetría de hombros: theta_hombros = arctan((y12 - y11)/(x12 - x11))
        2. Ladeo de cabeza (Head Roll): Ángulo entre nariz (1) y mentón (152) respecto a la vertical.
        """
        # 1. Alineación de hombros (Pose)
        angulo_hombros = 0.0
        detalle_hombros = "hombros_alineados"
        if pose_landmarks and len(pose_landmarks) > 0:
            p = pose_landmarks[0]
            if len(p) > max(SHOULDER_LEFT, SHOULDER_RIGHT):
                y11, x11 = p[SHOULDER_LEFT].y, p[SHOULDER_LEFT].x
                y12, x12 = p[SHOULDER_RIGHT].y, p[SHOULDER_RIGHT].x
                dx = abs(x12 - x11)
                dy = abs(y12 - y11)
                if dx > 1e-5:
                    angulo_hombros = math.degrees(math.atan2(dy, dx))
                if angulo_hombros > 5.0:
                    if y11 > y12:
                        detalle_hombros = "hombro_izquierdo_caido"
                    else:
                        detalle_hombros = "hombro_derecho_caido"

        # 2. Ladeo de cabeza (FaceMesh: Nariz -> Mentón)
        angulo_cabeza = 0.0
        detalle_cabeza = "cabeza_alineada"
        if face_landmarks:
            p_nariz = face_landmarks[NOSE_TIP]
            p_menton = face_landmarks[CHIN]
            dx_cabeza = p_menton.x - p_nariz.x
            dy_cabeza = p_menton.y - p_nariz.y
            # Ángulo respecto a la vertical (0, 1)
            angulo_cabeza = abs(math.degrees(math.atan2(dx_cabeza, dy_cabeza)))
            if angulo_cabeza > 8.0:
                detalle_cabeza = "cabeza_ladeada_derecha" if dx_cabeza > 0 else "cabeza_ladeada_izquierda"

        return angulo_hombros, angulo_cabeza, detalle_hombros, detalle_cabeza

    def _evaluar_tension_facial(
        self,
        blendshapes: Dict[str, float]
    ) -> Tuple[float, bool, List[str]]:
        """
        Eje 3: Microexpresiones y Tensión Facial
        - Tensión en el ceño (AU4 / Brow Down): promedio de browDownLeft y browDownRight
        - Presión labial (AU24 / Mouth Press): promedio de mouthPressLeft y mouthPressRight
        - Parpadeo: eyeBlinkLeft y eyeBlinkRight > 0.60
        """
        brow_down = (
            blendshapes.get("browDownLeft", 0.0) + blendshapes.get("browDownRight", 0.0)
        ) / 2.0
        mouth_press = (
            blendshapes.get("mouthPressLeft", 0.0) + blendshapes.get("mouthPressRight", 0.0)
        ) / 2.0

        indice_tension_frame = (brow_down + mouth_press) / 2.0

        blink_l = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_r = blendshapes.get("eyeBlinkRight", 0.0)
        es_parpadeo = (blink_l > 0.60 and blink_r > 0.60)

        detalles_tension = []
        if brow_down > 0.35:
            detalles_tension.append("ceño_fruncido_tension")
        if mouth_press > 0.35:
            detalles_tension.append("presion_labial_tension")
        if es_parpadeo:
            detalles_tension.append("parpadeo_ojos_cerrados")

        return indice_tension_frame, es_parpadeo, detalles_tension

    def _calcular_score_expresion(
        self,
        contacto_pct: float,
        hombros_deg: float,
        cabeza_deg: float,
        tension_idx: float,
        parpadeo_bpm: float,
        estabilidad_score: float
    ) -> float:
        """
        Calcula la rúbrica unificada del área de expresión (0 a 100).
        Ponderación:
        - Contacto visual: 30%
        - Postura y alineación: 25%
        - Tensión y serenidad: 25%
        - Estabilidad y presencia: 20%
        """
        score_contacto = contacto_pct

        penalizacion_hombros = max(0.0, hombros_deg - 5.0) * 3.0
        penalizacion_cabeza = max(0.0, cabeza_deg - 8.0) * 3.0
        score_postura = max(0.0, min(100.0, 100.0 - penalizacion_hombros - penalizacion_cabeza))

        penalizacion_tension = tension_idx * 100.0 * 0.75
        penalizacion_parpadeo = abs(20.0 - parpadeo_bpm) * 0.6
        score_tension = max(0.0, min(100.0, 100.0 - penalizacion_tension - penalizacion_parpadeo))

        score_estabilidad = estabilidad_score

        score_total = (
            (score_contacto * 0.30) +
            (score_postura * 0.25) +
            (score_tension * 0.25) +
            (score_estabilidad * 0.20)
        )

        return round(max(0.0, min(100.0, score_total)), 1)

    def procesar_frames(
        self,
        frames_input: Union[str, Path, List[Union[str, Path]]],
        fps: float = 0.5
    ) -> Dict:
        """
        Procesa una secuencia de imágenes (frames) o una carpeta de frames extraídos.
        Retorna las métricas cuantitativas globales junto con tuplas detalladas por cada fotograma.
        
        :param frames_input: Carpeta con archivos .jpg/.png o lista de rutas de imágenes.
        :param fps: Tasa de fotogramas procesada (por defecto 0.5 fps = 1 frame cada 2 seg).
        :return: Diccionario con la estructura de salida JSON y tuplas de eventos.
        """
        # Resolver lista de archivos
        if isinstance(frames_input, (str, Path)):
            p = Path(frames_input).resolve()
            if p.is_dir():
                archivos_frames = sorted(
                    [str(f) for f in p.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
                )
            elif p.is_file():
                archivos_frames = [str(p)]
            else:
                raise FileNotFoundError(f"Ruta no válida de frames: {frames_input}")
        elif isinstance(frames_input, list):
            archivos_frames = [str(Path(f).resolve()) for f in frames_input]
        else:
            raise ValueError("frames_input debe ser una ruta de carpeta o una lista de rutas.")

        total_frames = len(archivos_frames)
        if total_frames == 0:
            return {
                "total_frames_analizados": 0,
                "metricas_ejes": {
                    "contacto_visual_porcentaje": 0.0,
                    "desvios_mirada_total": 0,
                    "alineacion_hombros_grados_promedio": 0.0,
                    "ladeo_cabeza_grados_promedio": 0.0,
                    "indice_tension_facial": 0.0,
                    "tasa_parpadeo_por_minuto": 0.0,
                    "estabilidad_balanceo_score": 0.0
                },
                "score_area_expresion": 0.0,
                "tuplas_eventos_detectados": [],
                "desglose_tuplas_por_eje": {
                    "desvios_mirada": [],
                    "inclinacion_hombros": [],
                    "ladeo_cabeza": [],
                    "tension_facial": [],
                    "parpadeos": [],
                    "balanceo_lateral": []
                }
            }

        print(f"--> Analizando {total_frames} fotogramas (Intervalo: {fps} fps)...")

        frames_contacto_visual = 0
        hombros_grados_list = []
        cabeza_grados_list = []
        tension_facial_list = []
        parpadeos_totales = 0
        estaba_parpadeando = False
        centros_cara_x = []

        # Colección de tuplas de eventos
        tuplas_desvios_mirada: List[Tuple[str, str, str, Dict[str, float]]] = []
        tuplas_inclinacion_hombros: List[Tuple[str, str, float, str]] = []
        tuplas_ladeo_cabeza: List[Tuple[str, str, float, str]] = []
        tuplas_tension_facial: List[Tuple[str, str, float, List[str]]] = []
        tuplas_parpadeos: List[Tuple[str, str]] = []
        tuplas_balanceo: List[Tuple[str, str, float]] = []
        todas_las_tuplas: List[Tuple[Any, ...]] = []

        for idx, frame_path in enumerate(archivos_frames):
            frame_nombre = Path(frame_path).name
            img_bgr = cv2.imread(frame_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # 1. Detección Facial (Landmarks + Blendshapes)
            face_res = self.face_landmarker.detect(mp_image)
            # 2. Detección Postural (Pose)
            pose_res = self.pose_landmarker.detect(mp_image)

            face_landmarks = face_res.face_landmarks[0] if face_res.face_landmarks else None
            blendshapes = self._extraer_blendshapes_dict(face_res.face_blendshapes)
            pose_landmarks = pose_res.pose_landmarks if pose_res.pose_landmarks else None

            if face_landmarks:
                # Eje 1: Contacto visual
                mirada_directa, rx_l, rx_r, detalle_mirada = self._evaluar_contacto_visual(
                    face_landmarks, blendshapes
                )
                if mirada_directa:
                    frames_contacto_visual += 1
                else:
                    # Registrar tupla de desvío de mirada
                    tupla_mirada = (
                        frame_nombre,
                        "desvio_mirada",
                        detalle_mirada,
                        {"rx_left": round(rx_l, 2), "rx_right": round(rx_r, 2)}
                    )
                    tuplas_desvios_mirada.append(tupla_mirada)
                    todas_las_tuplas.append(tupla_mirada)

                # Eje 2: Ladeo de cabeza y simetría de hombros
                deg_hombros, deg_cabeza, det_hombros, det_cabeza = self._evaluar_alineacion(
                    face_landmarks, pose_landmarks
                )
                if deg_hombros > 0:
                    hombros_grados_list.append(deg_hombros)
                cabeza_grados_list.append(deg_cabeza)

                # Registrar tupla si hay desbalance o caída de hombros (> 5°)
                if deg_hombros > 5.0:
                    tupla_hombro = (
                        frame_nombre,
                        "inclinacion_hombros",
                        round(deg_hombros, 1),
                        det_hombros
                    )
                    tuplas_inclinacion_hombros.append(tupla_hombro)
                    todas_las_tuplas.append(tupla_hombro)

                # Registrar tupla si la cabeza está ladeada (> 8°)
                if deg_cabeza > 8.0:
                    tupla_cabeza = (
                        frame_nombre,
                        "ladeo_cabeza",
                        round(deg_cabeza, 1),
                        det_cabeza
                    )
                    tuplas_ladeo_cabeza.append(tupla_cabeza)
                    todas_las_tuplas.append(tupla_cabeza)

                # Eje 3: Tensión facial y parpadeos
                tension_frame, es_parpadeo, det_tension = self._evaluar_tension_facial(blendshapes)
                tension_facial_list.append(tension_frame)

                # Registrar tupla de tensión facial elevada (> 0.30 o gestos marcados)
                if tension_frame > 0.30 or ("ceño_fruncido_tension" in det_tension) or ("presion_labial_tension" in det_tension):
                    tupla_tension = (
                        frame_nombre,
                        "tension_facial",
                        round(tension_frame, 2),
                        det_tension
                    )
                    tuplas_tension_facial.append(tupla_tension)
                    todas_las_tuplas.append(tupla_tension)

                # Conteo de transiciones de parpadeo
                if es_parpadeo and not estaba_parpadeando:
                    parpadeos_totales += 1
                    tupla_parpadeo = (frame_nombre, "parpadeo")
                    tuplas_parpadeos.append(tupla_parpadeo)
                    todas_las_tuplas.append(tupla_parpadeo)
                estaba_parpadeando = es_parpadeo

                # Eje 4: Centro de la cara (coordenada X para balanceo lateral)
                centros_cara_x.append((frame_nombre, face_landmarks[NOSE_TIP].x))

        # -------------------------------------------------------------
        # Consolidación de Métricas
        # -------------------------------------------------------------
        pct_contacto = round((frames_contacto_visual / total_frames) * 100.0, 1)
        desvios_mirada = total_frames - frames_contacto_visual

        avg_hombros = round(float(np.mean(hombros_grados_list)), 1) if hombros_grados_list else 0.0
        avg_cabeza = round(float(np.mean(cabeza_grados_list)), 1) if cabeza_grados_list else 0.0
        avg_tension = round(float(np.mean(tension_facial_list)), 2) if tension_facial_list else 0.0

        duracion_segundos = total_frames * (1.0 / max(0.01, fps))
        duracion_minutos = max(0.01, duracion_segundos / 60.0)
        tasa_parpadeo = round(float(parpadeos_totales / duracion_minutos), 1)

        # Eje 4: Estabilidad y balanceo
        if len(centros_cara_x) > 1:
            valores_x = [cx for _, cx in centros_cara_x]
            sigma_x = float(np.std(valores_x))
            mean_x = float(np.mean(valores_x))
            score_estabilidad = round(max(0.0, min(100.0, 100.0 - (sigma_x * 400.0))), 1)

            # Detectar frames con balanceo / desviación notable del centro (> 2 * sigma o > 0.035)
            umbral_balanceo = max(0.035, 2.0 * sigma_x)
            for f_name, cx in centros_cara_x:
                diff = cx - mean_x
                if abs(diff) > umbral_balanceo:
                    direccion = "desplazamiento_derecha" if diff > 0 else "desplazamiento_izquierda"
                    tupla_bal = (f_name, "balanceo_lateral", round(diff, 3), direccion)
                    tuplas_balanceo.append(tupla_bal)
                    todas_las_tuplas.append(tupla_bal)
        else:
            score_estabilidad = 100.0

        score_expresion = self._calcular_score_expresion(
            contacto_pct=pct_contacto,
            hombros_deg=avg_hombros,
            cabeza_deg=avg_cabeza,
            tension_idx=avg_tension,
            parpadeo_bpm=tasa_parpadeo,
            estabilidad_score=score_estabilidad
        )

        resultado = {
            "total_frames_analizados": total_frames,
            "metricas_ejes": {
                "contacto_visual_porcentaje": pct_contacto,
                "desvios_mirada_total": desvios_mirada,
                "alineacion_hombros_grados_promedio": avg_hombros,
                "ladeo_cabeza_grados_promedio": avg_cabeza,
                "indice_tension_facial": avg_tension,
                "tasa_parpadeo_por_minuto": tasa_parpadeo,
                "estabilidad_balanceo_score": score_estabilidad
            },
            "score_area_expresion": score_expresion,
            "tuplas_eventos_detectados": todas_las_tuplas,
            "desglose_tuplas_por_eje": {
                "desvios_mirada": tuplas_desvios_mirada,
                "inclinacion_hombros": tuplas_inclinacion_hombros,
                "ladeo_cabeza": tuplas_ladeo_cabeza,
                "tension_facial": tuplas_tension_facial,
                "parpadeos": tuplas_parpadeos,
                "balanceo_lateral": tuplas_balanceo
            }
        }

        return resultado


def procesar_metricas_video(
    frames_path: Union[str, Path, List[Union[str, Path]]],
    fps: float = 0.5,
    output_json_path: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Función de entrada principal para evaluar los fotogramas generados por separar-audio-video.
    
    :param frames_path: Ruta a la carpeta de fotogramas, archivo o lista de rutas.
    :param fps: Frecuencia de muestreo de cuadros por segundo (por defecto 0.5 fps).
    :param output_json_path: Ruta opcional donde guardar el archivo .json con los resultados.
    :return: Payload estructurado con las métricas cuantitativas y tuplas de eventos por fotograma.
    """
    evaluador = EvaluadorExpresionVideo()
    resultado = evaluador.procesar_frames(frames_input=frames_path, fps=fps)

    if output_json_path:
        out_p = Path(output_json_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"[OK] Metricas de video guardadas en: {out_p}")

    return resultado


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta_carpeta_frames_o_frame.jpg> [fps]")
        print("Ejemplo: python main.py ../separar-audio-video/video_frames 0.5")
        sys.exit(1)

    input_path = sys.argv[1]
    input_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    resultado = procesar_metricas_video(frames_path=input_path, fps=input_fps)
    print("\n=== Resultado Evaluación de Video ===")
    print(f"Total frames: {resultado['total_frames_analizados']}")
    print(f"Score Expresion: {resultado['score_area_expresion']}")
    print(f"Metricas Ejes: {json.dumps(resultado['metricas_ejes'], indent=2, ensure_ascii=False)}")
    print(f"\nTotal tuplas de eventos: {len(resultado['tuplas_eventos_detectados'])}")
    for t in resultado['tuplas_eventos_detectados'][:10]:
        print(f" • {t}")
