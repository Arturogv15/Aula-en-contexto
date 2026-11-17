from dataclasses import dataclass, asdict
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class Segmento:
    id: int
    inicio: float   # segundos desde el comienzo del audio
    fin: float
    texto: str


def cargar_modelo(tamano: str = "large-v3", dispositivo: str = "cuda") -> WhisperModel:
    tipo = "float16" if dispositivo == "cuda" else "int8"
    return WhisperModel(tamano, device=dispositivo, compute_type=tipo)


def transcribir(modelo: WhisperModel, ruta_audio: Path, tamano: str = "large-v3") -> dict:
    segmentos, info = modelo.transcribe(
        str(ruta_audio),
        language="es",
        vad_filter=True,
    )
    lista = [
        Segmento(i, round(s.start, 2), round(s.end, 2), s.text.strip())
        for i, s in enumerate(segmentos)
    ]
    return {
        "audio": ruta_audio.name,
        "duracion": round(info.duration, 2),
        "modelo": tamano,
        "segmentos": [asdict(s) for s in lista],
    }