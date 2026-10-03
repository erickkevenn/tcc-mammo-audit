"""Analise de erro (S15), com tudo ja congelado e avaliado. Nada aqui muda modelo,
limiar ou resultado: so organiza os erros para inspecao.

Secoes (cada uma roda mesmo se outra falhar):
  massa       VinDr teste: falsos positivos e falsos negativos do detector de massas
              no limiar da regra V1 (0,4226). Acerto = centro da deteccao dentro da caixa.
  calc        o mesmo para microcalcificacoes (limiar 0,2692), na imagem preparada a 100 um.
  verificacao VinDr teste: alertas falsos nos pares originais e trocas nao encontradas,
              por regra e por categoria.
  v3          INbreast: os alertas da regra V3 nos laudos ORIGINAIS, com o achado que
              disparou. O trecho do laudo vai SO para o CSV local (artifacts/ nao vai
              para o git); o terminal mostra so identificador, categoria e achado.

Saidas em artifacts/erro/: <secao>_FP.csv, <secao>_FN.csv e pranchas PNG com os
recortes numerados (verde = caixa da base, vermelho = deteccao). A coluna 'causa'
fica vazia para ser preenchida na inspecao visual.

Uso:  python scripts/error_analysis.py              (todas)
      python scripts/error_analysis.py --only massa
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("artifacts/erro")
SEED = 20260819
N_SAMPLE = 30


# ------------------------------------------------------------------ casamento
def center_in(box, gt) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def match_errors(preds: pd.DataFrame, gts: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """preds: image_id, x1..y2, score; gts: image_id, x1..y2 (sem NaN).
    FN: caixa da base sem deteccao >= thr com centro dentro (guarda a maior pontuacao
    abaixo do limiar que caia nela). FP: deteccao >= thr com centro fora de toda caixa;
    so a de maior pontuacao por imagem."""
    P = {i: g for i, g in preds.groupby("image_id")}
    G = {i: g for i, g in gts.groupby("image_id")}
    fn, fp = [], []
    for iid, g in G.items():
        p = P.get(iid, preds.iloc[:0])
        for r in g.itertuples(index=False):
            gt = (r.x1, r.y1, r.x2, r.y2)
            inside = [s.score for s in p.itertuples(index=False) if center_in((s.x1, s.y1, s.x2, s.y2), gt)]
            if not any(sc >= thr for sc in inside):
                fn.append({"image_id": iid, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                           "melhor_score": max(inside, default=0.0)})
    for iid, p in P.items():
        g = G.get(iid)
        gl = [] if g is None else [(r.x1, r.y1, r.x2, r.y2) for r in g.itertuples(index=False)]
        bad = [s for s in p[p.score >= thr].itertuples(index=False)
               if not any(center_in((s.x1, s.y1, s.x2, s.y2), b) for b in gl)]
        if bad:
            s = max(bad, key=lambda t: t.score)
            fp.append({"image_id": iid, "x1": s.x1, "y1": s.y1, "x2": s.x2, "y2": s.y2, "score": s.score})
    fp = pd.DataFrame(fp, columns=["image_id", "x1", "y1", "x2", "y2", "score"]).sort_values("score", ascending=False)
    fn = pd.DataFrame(fn, columns=["image_id", "x1", "y1", "x2", "y2", "melhor_score"])
    return fp.reset_index(drop=True), fn


def pick(fp: pd.DataFrame, fn: pd.DataFrame, n: int = N_SAMPLE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FP: os n de maior pontuacao. FN: amostra aleatoria de n (semente fixa)."""
    fps = fp.head(n).reset_index(drop=True)
    fns = fn.sample(min(n, len(fn)), random_state=SEED).reset_index(drop=True) if len(fn) else fn
    return fps, fns


# ------------------------------------------------------------------ pranchas
def crop_tile(img8: np.ndarray, box, others, color, min_side: int, out: int = 200, label: str = ""):
    import cv2
    h, w = img8.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(min_side, 1.6 * (box[2] - box[0]), 1.6 * (box[3] - box[1]))
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    c = cv2.cvtColor(img8[y0:y1, x0:x1].copy(), cv2.COLOR_GRAY2BGR) if img8.ndim == 2 else img8[y0:y1, x0:x1].copy()
    for b, col in [(box, color)] + others:
        cv2.rectangle(c, (int(b[0] - x0), int(b[1] - y0)), (int(b[2] - x0), int(b[3] - y0)), col, 1)
    c = cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA)
    cv2.putText(c, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return c


def sheet(tiles: list, path: Path, cols: int = 6) -> None:
    import cv2
    if not tiles:
        return
    t = tiles[0].shape[0]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * t, cols * t, 3), np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * t:(r + 1) * t, c * t:(c + 1) * t] = im
    cv2.imwrite(str(path), canvas)


GREEN, RED = (0, 200, 0), (0, 0, 255)


def save_section(name, fps, fns, load_img, gts, preds, min_side):
    """CSV + prancha de FP e de FN. load_img(image_id) -> img8 nas coordenadas das caixas."""
    for kind, df, col in (("FP", fps, RED), ("FN", fns, GREEN)):
        df = df.copy()
        df.insert(0, "n", range(1, len(df) + 1))
        df["causa"] = ""
        df.to_csv(OUT / f"{name}_{kind}.csv", index=False)
        tiles = []
        for r in df.itertuples(index=False):
            try:
                img = load_img(r.image_id)
            except Exception as e:                       # imagem que nao abre: segue
                print(f"  {r.image_id}: {e}")
                continue
            box = (r.x1, r.y1, r.x2, r.y2)
            g = gts[gts.image_id == r.image_id]
            others = [((q.x1, q.y1, q.x2, q.y2), GREEN) for q in g.itertuples(index=False)]
            if kind == "FN":
                p = preds[(preds.image_id == r.image_id)].sort_values("score", ascending=False).head(3)
                others = [((q.x1, q.y1, q.x2, q.y2), RED) for q in p.itertuples(index=False)]
            score = getattr(r, "score", None) if kind == "FP" else r.melhor_score
            tiles.append(crop_tile(img, box, others, col, min_side, label=f"{r.n} {score:.2f}"))
        sheet(tiles, OUT / f"{name}_{kind}.png")
        print(f"  {kind}: {len(df)} casos -> {OUT / f'{name}_{kind}.csv'} e .png")


def thresholds() -> dict:
    return yaml.safe_load(Path("configs/verifier_thresholds.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ secoes
def section_massa() -> None:
    import cv2
    from src.config import paths
    from src.eval.breast_eval import yolo_to_xyxy
    thr = thresholds()["mass"]
    cache = Path(paths()["out"]["cache"]) / "yolo_B0_v2_neg1"
    man = pd.read_csv(cache / "split_manifest.csv")
    man = man[man.split == "test"]
    img_dir, lbl_dir = cache / "images" / "test", cache / "labels" / "test"
    rows = []
    for iid in man.image_id:
        f = lbl_dir / f"{iid}.txt"
        if not f.exists():
            continue
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        if lines:
            h, w = cv2.imread(str(img_dir / f"{iid}.png"), cv2.IMREAD_GRAYSCALE).shape
            for c, b in (yolo_to_xyxy(ln, w, h) for ln in lines):
                if c == 0:
                    rows.append({"image_id": iid, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
    gts = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2"])
    pr = pd.read_csv("artifacts/eval/det_mass_B0_v2_neg1_test_preds.csv")
    pr = pr[pr.cls == 0]
    fp, fn = match_errors(pr, gts, thr)
    print(f"massa (teste, limiar {thr:.4f}): {len(gts)} massas na base, {len(fn)} nao encontradas; "
          f"{len(fp)} imagens com falso positivo acima do limiar, de {len(man)}")
    fps, fns = pick(fp, fn)
    save_section("massa", fps, fns, lambda i: cv2.imread(str(img_dir / f"{i}.png"), cv2.IMREAD_GRAYSCALE),
                 gts, pr, min_side=96)


def section_calc() -> None:
    from src.config import load, paths
    from src.preprocess.vindr_calc_tiles import calc_table, prepare
    thr = thresholds()["calc"]
    cfg = paths()
    root = Path(cfg["root"]) / cfg["vindr"]["dir"]
    tab = calc_table(pd.read_csv(root / cfg["vindr"]["finding_csv"]))
    study = dict(zip(tab.image_id, tab.study_id))
    gts = pd.read_csv("artifacts/eval/det_calc_B0_test_gts.csv").dropna()
    pr = pd.read_csv("artifacts/eval/det_calc_B0_test_preds.csv")
    pr = pr[pr.score > 0]
    fp, fn = match_errors(pr, gts, thr)
    print(f"microcalcificacao (teste, limiar {thr:.4f}): {len(gts)} agrupamentos na base, {len(fn)} nao "
          f"encontrados; {len(fp)} imagens com falso positivo acima do limiar")
    fps, fns = pick(fp, fn)
    pre = load("preprocess.yaml")
    arm = pre["arms"]["B0"]
    cache = {}

    def load_img(iid):
        if iid not in cache:
            res = prepare(root / cfg["vindr"]["images"] / study[iid] / f"{iid}.dicom", arm, pre, 100.0)
            cache[iid] = res[0]
        return cache[iid]
    save_section("calc", fps, fns, load_img, gts, pr, min_side=256)


def section_verificacao() -> None:
    al = pd.read_csv("artifacts/verify/alertas_vindr_teste.csv", dtype={"categoria": str})
    main = ["V1", "V3", "V4"]
    al["principal"] = al[main].any(axis=1)
    orig = al[al.tipo == "original"]
    cat_unit = dict(zip(orig.unidade, orig.categoria))
    print("VinDr teste, pares ORIGINAIS: alerta (falso) por categoria e regra")
    t = orig.groupby("categoria").agg(n=("principal", "size"), alerta=("principal", "sum"),
                                       V1=("V1", "sum"), V4=("V4", "sum"), V2=("V2", "sum"))
    t["taxa"] = (t.alerta / t.n).round(3)
    print(t.to_string())
    sw = al[al.tipo == "trocado"].copy()
    sw["cat_imagem"] = sw.unidade.map(cat_unit)
    miss = sw[~sw.principal]
    print("\nPARES TROCADOS nao encontrados, por categoria da imagem e categoria recebida")
    print(pd.crosstab(miss.cat_imagem, miss.categoria, margins=True).to_string())
    print("\nrebaixadas: encontradas por categoria da imagem")
    r = sw[sw.direcao == "rebaixada"].groupby("cat_imagem").principal.agg(["size", "sum", "mean"]).round(3)
    print(r.to_string())
    t.to_csv(OUT / "verificacao_originais.csv")
    miss.to_csv(OUT / "verificacao_trocas_perdidas.csv", index=False)


def section_v3() -> None:
    from src.data.inbreast import load_reports
    from src.report.extract_rules import RuleExtractor
    from src.report.schema import Status
    al = pd.read_csv("artifacts/verify/alertas_inbreast_esc70.csv", dtype={"categoria": str})
    v3 = al[(al.tipo == "original") & al.V3]
    rep = load_reports().set_index("file")
    ext = RuleExtractor("pt")
    rows = []
    for r in v3.itertuples(index=False):
        text = rep.loc[r.fonte, "text"]
        for side in ("L", "R"):
            for f in ext.extract(text, r.fonte, side).findings:
                if f.status == Status.AFFIRMED and f.suspicion is not None:
                    rows.append({"laudo": r.fonte, "categoria": r.categoria, "lado": side,
                                 "achado": str(f.category.value if hasattr(f.category, "value") else f.category),
                                 "suspeicao": f.suspicion,
                                 "trecho_LOCAL": (f.evidence.sentence_text or "").strip()[:300],
                                 "causa": ""})
    df = pd.DataFrame(rows)
    print(f"INbreast: {len(v3)} laudos originais com alerta V3 (resultado congelado)")
    if len(df):
        print(df.drop(columns=["trecho_LOCAL", "causa"]).drop_duplicates().to_string(index=False))
    df.to_csv(OUT / "v3_inbreast_alertas.csv", index=False, encoding="utf-8-sig")
    print(f"-> {OUT / 'v3_inbreast_alertas.csv'} (tem trecho de laudo: so local, nao copiar para o texto)")


SECTIONS = {"massa": section_massa, "calc": section_calc,
            "verificacao": section_verificacao, "v3": section_v3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SECTIONS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fails = []
    for name, fn in SECTIONS.items():
        if args.only and name != args.only:
            continue
        print(f"\n===== {name} =====")
        try:
            fn()
        except Exception:
            traceback.print_exc()
            fails.append(name)
    print(f"\nsecoes com erro: {fails or 'nenhuma'}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
