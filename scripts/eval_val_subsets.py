"""Avalia um checkpoint na VALIDACAO de duas formas, para separar modelo de metrica.

    python scripts/eval_val_subsets.py --weights runs/detect/artifacts/runs/det_mass_B0_v2/weights/best.pt

1. validacao completa (positivas + normais): o numero honesto, com a prevalencia
   real, onde cada deteccao numa imagem normal conta como falso positivo;
2. so as imagens COM achado da validacao: comparavel ao jeito que o M1 era medido.

Se (2) ficar perto do M1 e (1) muito abaixo, o modelo nao piorou: o que pesa sao
os falsos positivos nas normais, e a alavanca e ensinar "fundo" (mais normais no
treino), nao mexer na arquitetura. Nunca roda no teste.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml


def _data_yaml(cache: Path, names: dict, image_list: Path, tag: str) -> Path:
    y = cache / f"_eval_{tag}.yaml"
    y.write_text(yaml.safe_dump({"path": str(cache), "train": str(image_list),
                                 "val": str(image_list), "names": names}), encoding="utf-8")
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", default="E:/Erick/TCC/tcc_cache/yolo_B0_v2")
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    from ultralytics import YOLO

    cache = Path(args.cache)
    names = yaml.safe_load((cache / "data.yaml").read_text(encoding="utf-8"))["names"]
    man = pd.read_csv(cache / "split_manifest.csv")
    val = man[man.split == "val"]

    subsets = {"completa": val, "so_positivas": val[val.role == "pos"]}
    model = YOLO(args.weights)
    linhas = []
    for tag, sub in subsets.items():
        lst = cache / f"_eval_{tag}.txt"
        lst.write_text("\n".join(str(cache / "images" / "val" / f"{i}.png") for i in sub.image_id),
                       encoding="utf-8")
        m = model.val(data=str(_data_yaml(cache, names, lst, tag)), split="val",
                      imgsz=args.imgsz, rect=True, batch=4, plots=False, verbose=False)
        row = {"subconjunto": tag, "imagens": len(sub), "mAP50_all": round(float(m.box.map50), 4)}
        for cls_i, ap50 in zip(m.ap_class_index, m.box.ap50):
            row[f"AP50_{names[int(cls_i)]}"] = round(float(ap50), 4)
        linhas.append(row)

    print("\n" + pd.DataFrame(linhas).set_index("subconjunto").T.to_string())


if __name__ == "__main__":
    main()
