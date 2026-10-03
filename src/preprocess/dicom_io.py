"""Leitura e janelamento de DICOM na ORDEM CORRETA do padrao DICOM PS3.3:

    Stored Values -> Modality LUT (Rescale) -> VOI LUT / janelamento -> polaridade

Inverter MONOCHROME1 ANTES do janelamento e o bug silencioso mais comum da area.
O codigo oficial do VinDr inverte DEPOIS -- copie esse comportamento.

Notas de armadilha:
  * cast para int32, NAO int16: int16 estoura em dados 16-bit unsigned.
  * apply_voi_lut do pydicom retorna float64 no caminho WC/WW e inteiro no
    caminho LUT -- encadear apply_modality_lut + apply_voi_lut quebra.
    Por isso implementamos o janelamento aqui.
  * numpy >= 1.24 removeu np.int/np.float: codigo antigo de mamografia quebra.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def read_dicom(path: str | Path) -> tuple[np.ndarray, dict]:
    """Retorna (array int32 apos Modality LUT, metadados relevantes)."""
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.int32)                     # int32, nao int16
    color_pm = None
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):             # DICOM colorido (DMID): vira cinza
        color_pm = str(getattr(ds, "PhotometricInterpretation", "RGB"))
        arr = to_gray(arr, color_pm)

    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = (arr * slope + inter).astype(np.float32)            # Modality LUT

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and hasattr(wc, "__len__") and not isinstance(wc, str):
        wc = float(wc[0])                                     # MultiValue -> indice 0
    if ww is not None and hasattr(ww, "__len__") and not isinstance(ww, str):
        ww = float(ww[0])

    meta = {
        "photometric": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
        "wc": None if wc is None else float(wc),
        "ww": None if ww is None else float(ww),
        "voi_fn": str(getattr(ds, "VOILUTFunction", "LINEAR") or "LINEAR").upper(),
        "laterality": getattr(ds, "ImageLaterality", None),
        "view": getattr(ds, "ViewPosition", None),
        "presentation_intent": str(getattr(ds, "PresentationIntentType", "") or ""),
        "padding_value": getattr(ds, "PixelPaddingValue", None),
        "bits_stored": getattr(ds, "BitsStored", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None),
        "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
    }
    if color_pm is not None:
        meta["photometric"] = "MONOCHROME2"
        meta["convertido_de"] = color_pm
    # Fallback documentado no Mirai: lateralidade pode vir dentro de ViewPosition.
    if not meta["laterality"] and isinstance(meta["view"], str):
        v = meta["view"].upper()
        if v.startswith("R"):
            meta["laterality"] = "R"
        elif v.startswith("L"):
            meta["laterality"] = "L"
    return arr, meta


def to_gray(arr: np.ndarray, photometric: str = "RGB") -> np.ndarray:
    """Imagem colorida (linhas x colunas x 3 ou 4) -> cinza por luminancia (ITU-R BT.601).

    Alguns DICOM publicos (DMID) gravam a mamografia como RGB de 8 bits. Se os tres
    canais forem iguais, o resultado e o proprio canal. YBR e convertido para RGB antes.
    """
    a = np.asarray(arr)[..., :3]
    if str(photometric).upper().startswith("YBR"):
        from pydicom.pixel_data_handlers.util import convert_color_space
        a = convert_color_space(a.astype(np.uint8), str(photometric).upper(), "RGB")
    a = a.astype(np.float32)
    return np.rint(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.int32)


def apply_windowing(arr: np.ndarray, wc: float, ww: float, fn: str = "LINEAR",
                    y_min: float = 0.0, y_max: float = 255.0) -> np.ndarray:
    """LINEAR / LINEAR_EXACT / SIGMOID conforme DICOM PS3.3 C.11.2.1.2."""
    arr = arr.astype(np.float32)
    rng = y_max - y_min
    if fn == "SIGMOID":
        return y_min + rng / (1.0 + np.exp(-4.0 * (arr - wc) / max(ww, 1e-6)))
    if fn == "LINEAR_EXACT":
        out = (arr - wc) / max(ww, 1e-6) * rng + y_min
        return np.clip(out, y_min, y_max)
    # LINEAR (padrao)
    lo = wc - 0.5 - (ww - 1) / 2.0
    hi = wc - 0.5 + (ww - 1) / 2.0
    out = ((arr - (wc - 0.5)) / max(ww - 1, 1e-6) + 0.5) * rng + y_min
    out[arr <= lo] = y_min
    out[arr > hi] = y_max
    return out


def winsor_scale(arr: np.ndarray, mask: np.ndarray | None = None,
                 low: float = 1.0, high: float = 99.0, y_max: float = 65535.0) -> np.ndarray:
    """Normalizacao robusta por percentis DENTRO da mascara da mama.

    Calcular percentis na imagem inteira deixa o fundo dominar. Winsor p0.5-p99.5
    restrita a mama elevou 'effective bits' de 7,88 -> 10,11 (Technologies, 2026).
    """
    vals = arr[mask] if mask is not None and mask.any() else arr
    lo, hi = np.percentile(vals, [low, high])
    out = (arr.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0) * y_max


def finalize_polarity(img: np.ndarray, photometric: str, vmax: float) -> np.ndarray:
    """Inversao MONOCHROME1 -- SEMPRE POR ULTIMO."""
    if photometric.upper() == "MONOCHROME1":
        return vmax - img
    return img
