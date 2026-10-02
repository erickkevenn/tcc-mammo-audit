"""Avaliacao final da verificacao da categoria BI-RADS (S14), nos pares CONGELADOS.

Para cada par (imagens de uma unidade + categoria/laudo de uma fonte), aplica as
regras V1 a V4 com os limiares congelados (configs/verifier_thresholds.yaml) e mede:
  * alerta falso nos pares originais (consistentes): especificidade;
  * inconsistencia encontrada nos pares trocados, por direcao:
      rebaixada (imagem 4/5 com categoria 1 a 3) e elevada (imagem 1 a 3 com 4/5);
  * acuracia balanceada = (especificidade + media das duas direcoes) / 2.
IC 95% por bootstrap de pacientes (INbreast) ou estudos (VinDr).

Metodos comparados (subconjuntos das regras, nos MESMOS pares):
  principal (V1+V3+V4)  resultado principal, decidido antes do teste
  principal + V2        V2 reportada a parte
  so imagem (V1+V4), so detectores (V1), so classificador (V4)
  so texto (V3)         so no INbreast (o VinDr nao tem texto)
McNemar pareado: principal contra so texto e contra so imagem.

Bases:
  vindr_val    validacao do VinDr, pares montados na hora (k=1). Serve para
               conferir o codigo: deve reproduzir a calibracao. Nao precisa de confirmacao.
  vindr_teste  pares congelados do teste oficial; exige --confirm-test.
  inbreast     pares congelados com laudos reais; exige --confirm-test. A unidade
               e o EXAME (o laudo da uma categoria por exame): a imagem do exame e a
               maior pontuacao entre as imagens e a maior P(4)+P(5) entre as mamas;
               os achados do laudo saem do extrator E0 congelado.

Saidas: artifacts/verify/resultado_<base>.json e alertas_<base>.csv (sem texto de laudo).

Uso:
  python scripts/evaluate_verification.py --base vindr_val
  python scripts/evaluate_verification.py --base vindr_teste --confirm-test
  python scripts/evaluate_verification.py --base inbreast --confirm-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verify.rules import (BreastImage, ReportFinding, Thresholds,     # noqa: E402
                              biopsy_side, verify_breast)

RULES = ["V1", "V2", "V3", "V4"]
METHODS = {
    "principal (V1+V3+V4)": ["V1", "V3", "V4"],
    "principal + V2": ["V1", "V2", "V3", "V4"],
    "so imagem (V1+V4)": ["V1", "V4"],
    "so detectores (V1)": ["V1"],
    "so classificador (V4)": ["V4"],
    "so texto (V3)": ["V3"],
}
CELLS = ["orig", "orig13", "orig45", "rebaixada", "elevada"]
MASS_RUN, CALC_RUN, CLS_RUN = "det_mass_B0_v2_neg1", "det_calc_B0", "cls_b2_imagenet"
SEED = 20260819


# ------------------------------------------------------------------ pontuacoes da imagem
def unit_images(img: pd.DataFrame, mass: pd.DataFrame, calc: pd.DataFrame,
                cls: pd.DataFrame) -> dict[str, BreastImage]:
    """img: image_id, unidade, mama. Massa e calcificacao: maior pontuacao entre as
    imagens da unidade (0 sem deteccao). P(4)+P(5): media das vistas de cada mama e,
    se a unidade tem duas mamas (exame), a maior delas."""
    if "cls" in mass:
        mass = mass[mass.cls == 0]
    m = mass.groupby("image_id").score.max()
    c = calc.groupby("image_id").score.max()
    t = img.assign(mass=img.image_id.map(m).fillna(0.0), calc=img.image_id.map(c).fillna(0.0))
    pc = cls.set_index("image_id")
    if not t.image_id.isin(pc.index).all():
        falta = int((~t.image_id.isin(pc.index)).sum())
        raise SystemExit(f"{falta} imagens sem predicao do classificador")
    t["p_ge4"] = t.image_id.map(pc.pb3 + pc.pb4)
    per_breast = t.groupby(["unidade", "mama"]).p_ge4.mean().groupby("unidade").max()
    agg = t.groupby("unidade").agg(mass=("mass", "max"), calc=("calc", "max"))
    return {u: BreastImage(float(r.mass), float(r.calc), float(per_breast[u])) for u, r in agg.iterrows()}


# ------------------------------------------------------------------ regras por par
def alerts_table(pairs: pd.DataFrame, images: dict[str, BreastImage],
                 findings: dict[str, list[ReportFinding]] | None, thr: Thresholds) -> pd.DataFrame:
    rows = []
    for p in pairs.itertuples(index=False):
        f = findings.get(p.fonte, []) if findings is not None else None
        codes = {a.rule for a in verify_breast(str(p.categoria), images[p.unidade], f, thr)}
        rows.append({r: r in codes for r in RULES})
    out = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["verificavel"] = out.categoria.astype(str).map(biopsy_side).notna()
    return out


def cell_masks(al: pd.DataFrame) -> dict[str, np.ndarray]:
    o, t, hi = (al.tipo == "original").to_numpy(), (al.tipo == "trocado").to_numpy(), (al.lado_imagem == 1).to_numpy()
    return {"orig": o, "orig13": o & ~hi, "orig45": o & hi, "rebaixada": t & hi, "elevada": t & ~hi}


def metrics_from(rate: dict[str, float]) -> dict[str, float]:
    esp = 1 - rate["orig"]
    media = (rate["rebaixada"] + rate["elevada"]) / 2
    return {"especificidade": esp, "alerta_originais_1a3": rate["orig13"], "alerta_originais_4e5": rate["orig45"],
            "encontra_rebaixada": rate["rebaixada"], "encontra_elevada": rate["elevada"],
            "media_direcoes": media, "acuracia_balanceada": (esp + media) / 2}


def evaluate(al: pd.DataFrame, methods: dict[str, list[str]], n_boot: int = 2000,
             seed: int = SEED) -> dict[str, dict]:
    """Ponto e IC 95% (bootstrap por grupo) de cada metrica, para cada metodo."""
    masks = cell_masks(al)
    groups, gidx = np.unique(al.grupo.to_numpy(), return_inverse=True)
    G = len(groups)
    # soma de alertas e contagem por grupo, por celula: reamostrar grupos = somar linhas
    cnt = np.stack([np.bincount(gidx, weights=masks[c], minlength=G) for c in CELLS], 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(n_boot, G))
    W = np.stack([np.bincount(d, minlength=G) for d in draws])        # vezes que cada grupo entra
    res = {}
    for name, rules in methods.items():
        hit = al[rules].any(axis=1).to_numpy()
        s = np.stack([np.bincount(gidx, weights=masks[c] & hit, minlength=G) for c in CELLS], 1)
        point = metrics_from({c: s[:, i].sum() / max(cnt[:, i].sum(), 1) for i, c in enumerate(CELLS)})
        S, N = W @ s, W @ cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            R = S / N
        boot = [metrics_from({c: R[b, i] for i, c in enumerate(CELLS)}) for b in range(n_boot)]
        bt = pd.DataFrame(boot)
        res[name] = {k: {"valor": float(v), "ic95": [float(np.nanpercentile(bt[k], 2.5)),
                                                    float(np.nanpercentile(bt[k], 97.5))]}
                     for k, v in point.items()}
        res[name]["n"] = {c: int(cnt[:, i].sum()) for i, c in enumerate(CELLS)}
    return res


def paired_tests(al: pd.DataFrame, a: str, b: str) -> dict:
    from src.eval.stats import mcnemar
    ok = {}
    for name in (a, b):
        hit = al[METHODS[name]].any(axis=1)
        ok[name] = np.where(al.tipo == "trocado", hit, ~hit)
    out = {}
    for sub, m in (("originais", al.tipo == "original"), ("trocados", al.tipo == "trocado"), ("todos", al.tipo.notna())):
        m = m.to_numpy()
        out[sub] = mcnemar(ok[a][m], ok[b][m])
    return out


# ------------------------------------------------------------------ bases
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_pairs(name: str) -> pd.DataFrame:
    path = Path(f"artifacts/pairs/{name}_pares.csv")
    frozen = yaml.safe_load(Path("configs/pairs_frozen.yaml").read_text(encoding="utf-8"))[name]
    h = sha256(path)
    if h != frozen["sha256"]:
        raise SystemExit(f"{path} nao e o conjunto congelado ({h[:12]} != {frozen['sha256'][:12]})")
    print(f"pares congelados conferidos: {path} (sha256 {h[:12]})")
    return pd.read_csv(path, dtype={"categoria": str})


def vindr_base(split: str):
    from src.config import dset, paths
    from src.data.vindr_breast import cls_table
    tab = cls_table(pd.read_csv(dset("vindr") / paths()["vindr"]["breast_csv"]))
    tab = tab[tab.split == split]
    img = pd.DataFrame({"image_id": tab.image_id, "unidade": tab.study_id + "_" + tab.laterality})
    img["mama"] = img.unidade
    ev = Path("artifacts/eval")
    mass = pd.read_csv(ev / f"{MASS_RUN}_{split}_preds.csv")
    calc = pd.read_csv(ev / f"{CALC_RUN}_{split}_preds.csv")
    cls = pd.read_csv(Path("artifacts/cls_runs") / CLS_RUN / f"preds_{split}.csv")
    seen = set(calc.image_id)
    if not img.image_id.isin(seen).all():
        raise SystemExit(f"{int((~img.image_id.isin(seen)).sum())} imagens sem predicao de microcalcificacao "
                         f"(a avaliacao do detector terminou?)")
    if split == "val":
        from scripts.build_pairs import make_pairs
        br = tab.groupby(["study_id", "laterality"]).birads.first().reset_index()
        cat = (br.birads + 1).astype(str)
        units = pd.DataFrame({"unidade": br.study_id + "_" + br.laterality, "grupo": br.study_id,
                              "lado": cat.map(biopsy_side).astype(int), "categoria": cat})
        pairs = make_pairs(units, k=1)
    else:
        pairs = load_frozen_pairs("vindr_teste")
    return pairs, unit_images(img, mass, calc, cls), None


def inbreast_base():
    from src.data.inbreast import load_index, load_reports
    from src.report.extract_rules import RuleExtractor
    from src.report.schema import Status
    pairs = load_frozen_pairs("inbreast")
    rep = load_reports()
    idx = load_index().rename(columns={"File Name": "acc", "Acquisition date": "date"})
    date = dict(zip(idx.acc.astype(str), idx.date.astype(str)))
    d = Path("artifacts/inbreast")
    mass, calc, cls = (pd.read_csv(d / f"{n}_preds.csv") for n in ("mass", "calc", "cls"))
    ids = pd.DataFrame({"image_id": cls.image_id})
    parts = ids.image_id.str.split("_")
    ids["acc"], ids["patient"], ids["lado"] = parts.str[0], parts.str[1], parts.str[3]
    rows = []
    for r in rep.itertuples(index=False):
        g = ids[ids.patient == r.patient]
        if isinstance(r.period, str) and r.period:
            g = g[g.acc.map(date) == r.period.replace("_", "")]
        rows += [{"image_id": i, "unidade": r.file, "mama": f"{r.file}_{s}"} for i, s in zip(g.image_id, g.lado)]
    img = pd.DataFrame(rows)
    falta = set(pairs.unidade) - set(img.unidade)
    if falta:
        raise SystemExit(f"{len(falta)} exames dos pares sem imagens ligadas")
    img = img[img.unidade.isin(set(pairs.unidade))]
    ext = RuleExtractor("pt")
    findings = {}
    for r in rep[rep.file.isin(set(pairs.fonte))].itertuples(index=False):
        fs = []
        for side in ("L", "R"):
            for f in ext.extract(r.text, r.file, side).findings:
                fs.append(ReportFinding(f.category, f.status == Status.AFFIRMED, f.suspicion))
        findings[r.file] = fs
    return pairs, unit_images(img, mass, calc, cls), findings


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=["vindr_val", "vindr_teste", "inbreast"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.base != "vindr_val" and not args.confirm_test:
        raise SystemExit("Avaliacao final: roda uma vez, com tudo congelado. Use --confirm-test se for isso.")

    cfg = yaml.safe_load(Path("configs/verifier_thresholds.yaml").read_text(encoding="utf-8"))
    thr = Thresholds(cfg["mass"], cfg["calc"], cfg["cls_ge4"])
    if args.base == "inbreast":
        pairs, images, findings = inbreast_base()
        methods = METHODS
    else:
        pairs, images, findings = vindr_base("val" if args.base == "vindr_val" else "test")
        methods = {k: v for k, v in METHODS.items() if k != "so texto (V3)"}

    al = alerts_table(pairs, images, findings, thr)
    if not al.verificavel.all():
        raise SystemExit(f"{int((~al.verificavel).sum())} pares com categoria nao verificavel")
    res = evaluate(al, methods, args.n_boot)

    print(f"\n{args.base} | limiares massa {thr.mass:.4f}, calcificacao {thr.calc:.4f}, P(4)+P(5) {thr.cls_ge4:.4f}")
    n = res[next(iter(res))]["n"]
    print(f"pares: originais {n['orig']} (1 a 3: {n['orig13']}, 4 e 5: {n['orig45']}) | "
          f"trocados rebaixada {n['rebaixada']}, elevada {n['elevada']}")
    rows = {m: {k: f"{v['valor']:.3f} [{v['ic95'][0]:.3f} a {v['ic95'][1]:.3f}]"
                for k, v in r.items() if k != "n"} for m, r in res.items()}
    pd.set_option("display.width", 250)
    print("\n" + pd.DataFrame(rows).T[["especificidade", "encontra_rebaixada", "encontra_elevada",
                                       "acuracia_balanceada"]].to_string())
    print("\nalertas falsos nos originais, por lado:")
    print(pd.DataFrame(rows).T[["alerta_originais_1a3", "alerta_originais_4e5"]].to_string())

    tests = {"principal x so imagem": paired_tests(al, "principal (V1+V3+V4)", "so imagem (V1+V4)")}
    if "so texto (V3)" in methods:
        tests["principal x so texto"] = paired_tests(al, "principal (V1+V3+V4)", "so texto (V3)")
    print("\nMcNemar (acerto = alerta no trocado, silencio no original):")
    for k, t in tests.items():
        for sub, r in t.items():
            print(f"  {k:24s} {sub:9s} so principal acerta {r['b_only_a_correct']:4d} | "
                  f"so o outro acerta {r['c_only_b_correct']:4d} | p = {r['p']:.4g} ({r['test']})")

    out = Path("artifacts/verify")
    out.mkdir(parents=True, exist_ok=True)
    al.to_csv(out / f"alertas_{args.base}.csv", index=False)
    (out / f"resultado_{args.base}.json").write_text(json.dumps(
        {"base": args.base, "limiares": cfg, "n_boot": args.n_boot, "metodos": res, "mcnemar": tests},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultado -> {out / f'resultado_{args.base}.json'}")


if __name__ == "__main__":
    main()
