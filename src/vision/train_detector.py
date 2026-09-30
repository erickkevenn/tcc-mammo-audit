"""Treino do detector. Perfil para <= 8 GB de VRAM.

    python -m src.vision.train_detector --data D:/Erick/tcc_cache/yolo_B0/data.yaml
    python -m src.vision.train_detector --config configs/detector_calc.yaml ...

Se estourar VRAM: imgsz 896x576, batch 1, accumulate 16. Se sobrar memoria,
SUBA A RESOLUCAO antes de trocar de modelo -- o retorno e maior (meia resolucao
custa -0,039 AUC no VinDr).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml gerado por to_yolo")
    ap.add_argument("--config", default="configs/detector_mass.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    imgsz = cfg.get("imgsz", [1024, 640])
    model = YOLO(f"{cfg['model']}.pt")
    model.train(
        data=args.data,
        imgsz=max(imgsz) if isinstance(imgsz, list) else imgsz,
        rect=True,                       # respeita a razao de aspecto da mama
        epochs=args.epochs or cfg.get("epochs", 80),
        batch=cfg.get("batch", 2),
        nbs=cfg.get("batch", 2) * cfg.get("accumulate", 8),   # batch nominal (acumulacao)
        amp=cfg.get("amp", True),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        cos_lr=cfg.get("cos_lr", True),
        warmup_epochs=cfg.get("warmup_epochs", 3),
        patience=cfg.get("patience", 15),
        workers=cfg.get("workers", 4),
        cache=cfg.get("cache", False),
        # Augmentation: geometrico leve + intensidade. SEM mixup/cutmix (sem
        # evidencia em mamografia); SEM flipud (anatomicamente implausivel e
        # desfaz a canonicalizacao de lateralidade).
        fliplr=0.5, flipud=0.0, mixup=0.0, copy_paste=0.0,
        degrees=15, translate=0.05, scale=0.15, shear=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        project="artifacts/runs",
        name=args.name or f"det_{Path(args.config).stem}",
        seed=20260819, deterministic=True,
    )


if __name__ == "__main__":
    main()
