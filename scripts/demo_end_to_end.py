"""Demonstracao da fatia vertical (marco M1 da semana 3).

Le um laudo real do INbreast (portugues), roda o extrator de regras, e roda o
detector de imagem -- o TREINADO de verdade (YOLOv8s, se houver checkpoint em
--weights) ou uma saida simulada (fake_detector_output) como fallback, para
continuar funcionando sem GPU / antes do treino existir ou se nao achar imagem
DICOM pro exame pedido.

    export TCC_DATA_ROOT=/caminho/para/Dataset      # ou ajuste configs/paths.yaml
    python scripts/demo_end_to_end.py --side R
    python scripts/demo_end_to_end.py --side R --weights runs/detect/artifacts/runs/det_detector_mass/weights/best.pt

AVISO: o detector foi treinado em VinDr (multi-fabricante). Rodar ele no
INbreast (Siemens MammoNovation) e um teste de generalizacao de verdade -- e
esperado que as deteccoes sejam mais fracas/esparsas aqui do que no proprio
conjunto de teste do VinDr. Isso e uma previa do "teste externo" da S14, nao
um bug.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.report_html import render
from src.audit.rules import Auditor
from src.config import load as load_cfg
from src.data.inbreast import load_dicom_index, load_reports
from src.eval.error_injection import build_benchmark
from src.report.extract_rules import RuleExtractor
from src.report.schema import (Descriptors, Evidence, Finding, FindingRecord, Region)


def fake_detector_output(exam_id: str, laterality: str) -> FindingRecord:
    """Saida simulada -- usada so quando nao ha checkpoint treinado disponivel
    (--weights nao existe) ou nenhuma imagem DICOM foi encontrada pro exame."""
    return FindingRecord(
        exam_id=exam_id, laterality=laterality, source="image", extractor="cnn",
        breast_birads="5", breast_density="C",
        findings=[
            Finding(category="mass", confidence=0.91, birads="5",
                    region=Region(quadrant="UOQ", vertical="upper", horizontal="outer"),
                    descriptors=Descriptors(mass_margin="spiculated", mass_shape="irregular"),
                    evidence=Evidence(image_id=f"{exam_id}_{laterality}_DM_CC",
                                      bbox=[820.0, 240.0, 960.0, 380.0])),
            Finding(category="architectural_distortion", confidence=0.55,
                    region=Region(quadrant="UOQ"),
                    evidence=Evidence(image_id=f"{exam_id}_{laterality}_DM_MLO",
                                      bbox=[700.0, 300.0, 810.0, 410.0])),
        ],
    )


def _preprocess_for_detector(dcm_path: str, pre: dict, arm_cfg: dict):
    """Mesmo pipeline usado no treino (to_yolo.py / to_png.py): janela/winsor ->
    recorte da mama -> polaridade -> flip canonico -> resize+pad. Devolve um
    array uint8 pronto pra entrar no YOLO (mesma distribuicao do treino)."""
    import numpy as np
    from src.preprocess.breast_roi import breast_bbox, breast_mask, canonical_flip, resize_and_pad
    from src.preprocess.dicom_io import apply_windowing, finalize_polarity, read_dicom, winsor_scale

    arr, meta = read_dicom(dcm_path)
    mask = breast_mask(arr)
    if arm_cfg["crop_breast"]:
        bb = breast_bbox(arr)
        if bb is not None:
            x0, y0, x1, y1 = bb
            arr, mask = arr[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    if arm_cfg["intensity"] == "winsor":
        img = winsor_scale(arr, mask, pre["winsor"]["low"], pre["winsor"]["high"], 255.0)
    elif meta["wc"] is not None and meta["ww"]:
        img = apply_windowing(arr, meta["wc"], meta["ww"], meta["voi_fn"], 0, 255)
    else:
        img = winsor_scale(arr, mask, 1.0, 99.0, 255.0)

    img = finalize_polarity(img, meta["photometric"], 255.0)
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    if pre.get("canonical_laterality"):
        img8, _ = canonical_flip(img8, meta["laterality"])
    th, tw = pre["mass_size"]
    img8, _ = resize_and_pad(img8, th, tw)
    return img8


def real_detector_output(exam_id: str, laterality: str, weights: str, conf: float = 0.25):
    """Roda o YOLOv8s treinado (--weights) nas vistas do INbreast pro exame+lado
    dado. Devolve None se nao achar checkpoint, imagem ou o ultralytics -- nesses
    casos o chamador (main) cai pro fake_detector_output."""
    if not Path(weights).exists():
        print(f"aviso: checkpoint '{weights}' nao encontrado -- usando saida simulada.")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("aviso: pacote 'ultralytics' nao instalado -- usando saida simulada.")
        return None

    idx = load_dicom_index()
    rows = idx[(idx.patient == exam_id) & (idx.side == laterality)]
    if rows.empty:
        print(f"aviso: nenhuma imagem DICOM do INbreast para {exam_id} lado {laterality} "
              f"-- usando saida simulada.")
        return None

    pre = load_cfg("preprocess.yaml")
    arm_cfg = pre["arms"][pre["default_arm"]]
    classes = load_cfg("detector_mass.yaml")["classes"]

    import cv2
    model = YOLO(weights)
    cache_dir = Path("artifacts/demo_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dets_by_image, image_meta = {}, {}
    for _, r in rows.iterrows():
        img8 = _preprocess_for_detector(r["path"], pre, arm_cfg)
        png_path = cache_dir / f"{exam_id}_{laterality}_{r['view']}.png"
        cv2.imwrite(str(png_path), img8)

        results = model.predict(source=str(png_path), conf=conf, verbose=False)
        dets = []
        for box in results[0].boxes:
            cls_idx = int(box.cls[0])
            if cls_idx < len(classes):
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                dets.append({"category": classes[cls_idx], "score": float(box.conf[0]),
                             "bbox": [x1, y1, x2, y2]})
        image_id = f"{exam_id}_{laterality}_{r['view']}"
        dets_by_image[image_id] = dets
        image_meta[image_id] = {"exam_id": exam_id, "laterality": laterality, "view": r["view"]}

    from src.vision.aggregate import aggregate_detections
    merged = aggregate_detections(dets_by_image, image_meta, mode="max")
    dets = merged.get(f"{exam_id}_{laterality}", [])

    print(f"detector real: {len(dets)} deteccao(oes) acima de conf={conf} em "
          f"{len(rows)} vista(s) ({', '.join(rows['view'])}).")
    findings = [
        Finding(category=d["category"], confidence=d["score"],
                evidence=Evidence(image_id=d["image_id"], bbox=d["bbox"]))
        for d in dets
    ]
    return FindingRecord(exam_id=exam_id, laterality=laterality, source="image",
                         extractor="cnn", findings=findings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=None, help="patient id do INbreast; default: primeiro laudo disponivel")
    ap.add_argument("--side", default="R", choices=["L", "R"])
    ap.add_argument("--out", default="artifacts/demo_audit.html")
    ap.add_argument("--weights", default="runs/detect/artifacts/runs/det_detector_mass/weights/best.pt",
                     help="checkpoint do YOLOv8s treinado; se nao existir, usa saida simulada")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    reports = load_reports()
    if reports.empty:
        raise SystemExit("Nenhum laudo do INbreast encontrado -- confira TCC_DATA_ROOT / configs/paths.yaml.")
    row = (reports[reports.patient == args.exam].iloc[0]
           if args.exam and (reports.patient == args.exam).any()
           else reports.iloc[0])
    exam_id, text = row.patient, row.text

    report_rec = RuleExtractor("pt").extract(text, exam_id, args.side)

    image_rec = real_detector_output(exam_id, args.side, args.weights, args.conf)
    if image_rec is None:
        image_rec = fake_detector_output(exam_id, args.side)

    auditor = Auditor()
    alerts = auditor.audit(image_rec, report_rec)
    alerts += auditor.audit_report_internal(report_rec)

    print("=" * 72)
    print(f"LAUDO EXTRAIDO ({exam_id}, mama {args.side})")
    print(json.dumps(report_rec.model_dump(exclude_none=True), indent=2, ensure_ascii=False)[:1500])
    print("=" * 72)
    print(f"{len(alerts)} ALERTA(S)\n")
    for a in alerts:
        print(f"  [{a.code}] sev={a.severity}  {a.message}")

    path = render(exam_id, alerts, text, None, args.out)
    print(f"\nrelatorio HTML -> {path}")

    bench = build_benchmark({exam_id: text}, language="pt", per_report=3)
    print(f"\nbenchmark sintetico gerado: {len(bench)} casos")
    for case in bench:
        print(f"  kind={case['kind']:<16} alerta esperado={case['expected_alert']}")


if __name__ == "__main__":
    main()