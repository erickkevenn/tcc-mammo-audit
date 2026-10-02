"""Conversao CBIS-DDSM -> dataset YOLO (cache em disco).

Complemento do to_yolo.py, que so sabe ler o VinDr. Aqui a caixa NAO vem de um
CSV: o CBIS-DDSM so distribui a mascara binaria da lesao, entao a bbox e
derivada da propria mascara.

Tres armadilhas do dataset, todas tratadas:
 1. As colunas 'cropped image file path' e 'ROI mask file path' TROCAM de lugar
    de forma inconsistente. Por isso a pasta '..._N' e resolvida por CONTEUDO
    com src.data.cbis.resolve_cropped_and_mask().
 2. A mascara as vezes nao tem exatamente as dimensoes da imagem completa. A
    bbox e reescalada proporcionalmente e a razao fica registrada no report.
 3. Os splits oficiais vazam paciente entre massa e calcificacao (13 + 18
    pacientes). Com --fix-leakage, todo paciente presente nos dois lados vai
    inteiro para o teste.

Uso:
    python -m src.preprocess.cbis_to_yolo --arm B0 --limit 50
    python -m src.preprocess.cbis_to_yolo --arm B0 --kind calc --config detector_calc.yaml
    python -m src.preprocess.cbis_to_yolo --arm B0 --fix-leakage

Saida (em <cache>/yolo_cbis_<mass|calc>_<arm>/):
    images/{training,test}/<series_base>.png
    labels/{training,test}/<series_base>.txt
    data.yaml
    report.csv   -- 1 linha por imagem: n_boxes, crop_area_pct, motivo do skip

Lembrete do plano (secao 1 do PLANO_TCC): CBIS-DDSM e filme digitalizado, nao
FFDM. Este cache serve para pre-treino e teste externo de robustez, nunca como
treino principal de um sistema FFDM.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import dset, load, paths
from ..data.cbis import load_all, resolve_cropped_and_mask
from .to_yolo import process_one

# tipo de anormalidade do CSV -> nomes de classe aceitos, na ordem de preferencia.
# O nome real vem do config do detector: detector_mass.yaml nao tem classe de
# calcificacao (o caminho de calcificacao e o detector_calc.yaml, por tiles),
# entao com o config de massa as linhas de calcificacao sao puladas e contadas
# no report, em vez de entrarem com uma classe inventada.
CLASS_CANDIDATES = {
    "mass": ["mass"],
    "calcification": ["calcification", "suspicious_calcification",
                      "microcalcification", "calc"],
}


def build_class_map(classes: list[str]) -> dict[str, str]:
    """Escolhe, para cada tipo do CSV, a primeira classe que existe no config."""
    out = {}
    for kind, candidates in CLASS_CANDIDATES.items():
        for c in candidates:
            if c in classes:
                out[kind] = c
                break
    return out


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Menor retangulo que contem a lesao. Devolve None se a mascara for vazia."""
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def scale_bbox(box, from_shape, to_shape):
    """Reescala a bbox quando a mascara nao tem o tamanho da imagem completa."""
    (h_from, w_from), (h_to, w_to) = from_shape, to_shape
    if (h_from, w_from) == (h_to, w_to):
        return box
    sx, sy = w_to / float(w_from), h_to / float(h_from)
    x1, y1, x2, y2 = box
    return int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))


def build_series_index(root: Path) -> dict[str, Path]:
    """Mapeia nome da serie ('Mass-Training_P_00001_LEFT_CC[_1]') -> pasta.

    O CBIS-DDSM guarda cada serie em <root>/<nome da serie>/<StudyUID>/<SeriesUID>/.
    Varrer uma vez e mais barato do que fazer rglob por linha do CSV.
    """
    index: dict[str, Path] = {}
    for d in root.rglob("*"):
        if d.is_dir() and ("_LEFT_" in d.name or "_RIGHT_" in d.name):
            index.setdefault(d.name, d)
    return index


def first_dicom(folder: Path) -> Path | None:
    files = sorted(folder.rglob("*.dcm"))
    return files[0] if files else None


def read_pixels(path: Path) -> np.ndarray:
    import pydicom
    return pydicom.dcmread(str(path)).pixel_array


def image_shape(path: Path) -> tuple[int, int]:
    """Altura e largura pelo cabecalho, sem decodificar os pixels.

    A imagem completa do CBIS-DDSM tem dezenas de megapixels; ler o array so
    para saber o tamanho dobra o custo de cada linha do CSV.
    """
    import pydicom
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    return int(ds.Rows), int(ds.Columns)


def collect_boxes(group: pd.DataFrame, index: dict[str, Path], full_shape,
                  cls_idx: dict[str, int], class_map: dict[str, str]) -> tuple[list, list[str]]:
    """Uma caixa por anormalidade da imagem, extraida da mascara binaria."""
    boxes, notes = [], []
    for _, r in group.iterrows():
        cls_name = class_map.get(str(r["abnormality_type"]).strip().lower())
        if cls_name is None or cls_name not in cls_idx:
            notes.append(f"classe fora do config: {r['abnormality_type']}")
            continue
        folder = index.get(r["series_roi"])
        if folder is None:
            notes.append(f"pasta ROI ausente: {r['series_roi']}")
            continue
        resolved = resolve_cropped_and_mask(folder)
        if resolved["mask"] is None:
            notes.append(f"mascara nao identificada: {r['series_roi']}")
            continue
        mask = read_pixels(resolved["mask"])
        bb = mask_bbox(mask)
        if bb is None:
            notes.append(f"mascara vazia: {r['series_roi']}")
            continue
        bb = scale_bbox(bb, mask.shape[:2], full_shape)
        boxes.append((bb[0], bb[1], bb[2], bb[3], cls_idx[cls_name]))
    return boxes, notes


def resolve_split(df: pd.DataFrame, fix_leakage: bool) -> pd.DataFrame:
    """Traduz official_split para o nome de pasta do YOLO, opcionalmente sem vazamento."""
    df = df.copy()
    df["split"] = df["official_split"].map({"train": "training", "test": "test"})
    if fix_leakage:
        both = (set(df[df.split == "training"].patient_id)
                & set(df[df.split == "test"].patient_id))
        df.loc[df.patient_id.isin(both), "split"] = "test"
        print(f">>> {len(both)} pacientes estavam nos dois splits; movidos inteiros para o teste.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="B0..B4 (default: preprocess.yaml)")
    ap.add_argument("--split", default=None, choices=["training", "test"])
    ap.add_argument("--kind", default=None, choices=["mass", "calc"])
    ap.add_argument("--limit", type=int, default=0, help="0 = todas as imagens")
    ap.add_argument("--fix-leakage", action="store_true",
                    help="paciente presente em treino e teste vai inteiro para o teste")
    ap.add_argument("--config", default="detector_mass.yaml",
                    help="config do detector que define a lista de classes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2

    pre = load("preprocess.yaml")
    arm = args.arm or pre["default_arm"]
    arm_cfg = pre["arms"][arm]
    classes = load(args.config)["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    class_map = build_class_map(classes)
    if not class_map:
        raise SystemExit(f"nenhum tipo do CBIS casa com as classes de {args.config}: {classes}")
    print(f"classes de {args.config}: {classes}")
    print(f"mapeamento CBIS -> classe: {class_map}")

    df = load_all()
    df = resolve_split(df, args.fix_leakage)
    if args.split:
        df = df[df.split == args.split]
    if args.kind:
        df = df[df.kind == args.kind]

    root = dset("cbis")
    print(f"indexando series em {root} ...")
    index = build_series_index(root)
    print(f"{len(index)} series encontradas")

    tag = Path(args.config).stem.replace("detector_", "")
    out_root = Path(args.out or f"{paths()['out']['cache']}/yolo_cbis_{tag}_{arm}")
    grouped = list(df.groupby(["patient_id", "series_base"]))
    if args.limit:
        grouped = grouped[: args.limit]

    rows: list[dict] = []
    n_ok = n_skip = 0
    for (patient_id, series_base), g in tqdm(grouped, desc=f"cbis {arm}"):
        row = {"patient_id": patient_id, "series": series_base,
               "split": str(g.iloc[0]["split"]), "kind": str(g.iloc[0]["kind"])}
        folder = index.get(series_base)
        dcm = first_dicom(folder) if folder else None
        if dcm is None:
            row["skipped"] = "imagem completa nao encontrada"
            rows.append(row); n_skip += 1
            continue
        try:
            full_shape = image_shape(dcm)
            boxes, notes = collect_boxes(g, index, full_shape, cls_idx, class_map)
            row["notes"] = "; ".join(notes) if notes else ""
            if not boxes:
                row["skipped"] = "nenhuma caixa valida"
                rows.append(row); n_skip += 1
                continue
            res = process_one(dcm, boxes, arm_cfg, pre,
                              laterality=str(g.iloc[0]["laterality"]))
        except Exception as exc:                                     # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row); n_skip += 1
            continue
        if res is None:
            row["skipped"] = "FOR PROCESSING"
            rows.append(row); n_skip += 1
            continue
        img, mapped, _ = res
        if not mapped:
            row["skipped"] = "caixas cairam fora do quadro apos o recorte"
            rows.append(row); n_skip += 1
            continue

        split = row["split"]
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "images" / split / f"{series_base}.png"), img)
        with open(out_root / "labels" / split / f"{series_base}.txt", "w") as fh:
            for cls, cx, cy, bw, bh in mapped:
                fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        row["n_boxes"] = len(mapped)
        rows.append(row); n_ok += 1

    yaml_txt = (f"path: {out_root}\ntrain: images/training\nval: images/test\n"
                f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data.yaml").write_text(yaml_txt, encoding="utf-8")

    rep = pd.DataFrame(rows)
    report = out_root / "report.csv"
    rep.to_csv(report, index=False)

    print(f"\n{n_ok} imagens escritas, {n_skip} ignoradas -> {out_root}")
    if "skipped" in rep:
        print("motivos de skip:")
        print(rep["skipped"].value_counts(dropna=True).to_string())
    if "error" in rep:
        print(f">>> {int(rep['error'].notna().sum())} erros de leitura (coluna 'error' do report).")
    if "n_boxes" in rep:
        print(f">>> {int(rep['n_boxes'].fillna(0).sum())} caixas escritas no total.")
    print(f"relatorio completo em {report}")
    print("Antes de treinar: rode pytest tests/test_bbox_mapping.py e abra 5 PNGs "
          "com o rotulo desenhado; caixa vinda de mascara e onde o erro passa silencioso.")


if __name__ == "__main__":
    main()