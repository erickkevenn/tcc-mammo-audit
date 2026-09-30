"""Injecao sintetica de erro em laudos REAIS -- camada 2 do benchmark.

Desenho baseado em ReXErr (12 tipos, PSB 2025) e CorBenchX (5 categorias,
26.326 laudos), adaptado ao lexico BI-RADS de mama.

REGRAS QUE NAO PODEM SER VIOLADAS:
 1. Perturbacao CONDICIONADA AO CONTEXTO (regra do ReXErr): nao injete
    'troca de margem de massa' num laudo sem massa. Erro impossivel e ruido,
    nao avaliacao.
 2. Injete em laudo REAL. Nunca gere o laudo a partir do rotulo -- isso torna
    a tarefa circular e trivialmente solucionavel.
 3. Se usar LLM para injetar E para detectar, use modelos e prompts DIFERENTES
    e declare isso no texto (vies de geracao).
 4. Validacao de plausibilidade: >= 100 pares revisados. O ReXErr usou 1 revisor
    e obteve 83/100 plausiveis. Use 2 avaliadores e reporte kappa de Cohen --
    e barato e supera o precedente.

Cada perturbacao devolve (texto_modificado, rotulo) com rotulo em nivel de
sentenca, para permitir metrica por tipo de erro.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass

# Analogo na literatura -> nosso codigo de alerta esperado
PERTURBATIONS = {
    "omission":        {"alert": "A1", "ref": "Omission (CorBenchX, 6.267 casos)"},
    "side_swap":       {"alert": "A3", "ref": "Side Confusion (CorBenchX, 7.615 -- a maior)"},
    "severity_change": {"alert": "A6", "ref": "Change Severity (ReXErr)"},
    "location_change": {"alert": "A4", "ref": "Change Location (ReXErr)"},
    "fabricated":      {"alert": "A2", "ref": "False Prediction (ReXErr)"},
    "descriptor_swap": {"alert": "A5", "ref": "-- diferencial deste TCC"},
    "density_change":  {"alert": "A7", "ref": "--"},
}

DESCRIPTOR_SWAPS = {
    "spiculated": "circumscribed", "speculated": "circumscribed",
    "circumscribed": "spiculated", "irregular": "round", "round": "irregular",
    "pleomorphic": "punctate", "punctate": "pleomorphic",
    "indistinct": "well defined",
    "espiculado": "circunscrito", "circunscrito": "espiculado",
    "irregular ": "arredondado ", "pleomorficas": "punctiformes",
}
SEVERITY_SWAPS = {"5": "4a", "4c": "3", "4b": "3", "4a": "2", "4": "3", "3": "2"}
LOCATION_SWAPS = {
    "upper": "lower", "lower": "upper", "outer": "inner", "inner": "outer",
    "superior": "inferior", "inferior": "superior",
    "externo": "interno", "interno": "externo",
}
DENSITY_SWAPS = {"ACR C": "ACR A", "ACR D": "ACR B", "ACR A": "ACR D", "ACR B": "ACR C"}
FABRICATED_SENTENCES = {
    "en": "An irregular spiculated mass is seen in the upper outer quadrant.",
    "pt": "Observa-se nodulo espiculado de contornos irregulares no quadrante superior externo.",
}


@dataclass
class Injection:
    kind: str
    text: str
    original_sentence: str | None
    modified_sentence: str | None
    expected_alert: str
    span: tuple[int, int] | None


def _sentences(text: str) -> list[tuple[str, int]]:
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            out.append((line, pos))
        pos += len(line)
    return out


def _swap_first(text: str, table: dict[str, str]) -> tuple[str, str, str] | None:
    for src, dst in table.items():
        m = re.search(re.escape(src), text, re.IGNORECASE)
        if m:
            return text[:m.start()] + dst + text[m.end():], src, dst
    return None


def applicable(text: str, kind: str) -> bool:
    """Condicionamento ao contexto -- checagem obrigatoria antes de injetar."""
    low = text.lower()
    if kind == "omission":
        return any(k in low for k in ("mass", "distortion", "calcification", "asymmetry",
                                      "nodulo", "nódulo", "distorcao", "microcalcif"))
    if kind == "side_swap":
        return ("left" in low and "right" in low) or ("esquerda" in low or "direita" in low)
    if kind == "severity_change":
        return re.search(r"bi[\-\s]?rads?", low) is not None
    if kind == "location_change":
        return any(k in low for k in LOCATION_SWAPS)
    if kind == "descriptor_swap":
        return any(k in low for k in DESCRIPTOR_SWAPS)
    if kind == "density_change":
        return "acr" in low
    if kind == "fabricated":
        return True
    return False


def inject(text: str, kind: str, language: str = "en",
           rng: random.Random | None = None) -> Injection | None:
    rng = rng or random.Random(20260819)
    if not applicable(text, kind):
        return None
    exp = PERTURBATIONS[kind]["alert"]

    if kind == "omission":
        # Omissao real = o achado deixa de ser mencionado EM QUALQUER LUGAR do laudo.
        # Remover apenas uma das sentencas (ex.: a de Findings, deixando a de Opinion)
        # produziria um laudo inconsistente, nao um laudo com omissao.
        terms = ("mass", "distortion", "calcification", "asymmetry",
                 "nodulo", "nódulo", "distorcao", "distorção", "microcalcif")
        present = [t for t in terms if t in text.lower()]
        if not present:
            return None
        # Tenta os termos em ordem aleatoria: um termo que so aparece em sentenca
        # NEGADA nao gera omissao (nao havia achado afirmado a omitir).
        for term in rng.sample(present, len(present)):
            removed, kept = [], []
            for sent, _pos in _sentences(text):
                neg = re.match(r"^\s*(no|sem|nao|não)\b", sent.strip(), re.IGNORECASE)
                if term in sent.lower() and not neg:
                    removed.append(sent)
                else:
                    kept.append(sent)
            if removed:
                return Injection(kind, "".join(kept), "".join(removed), None, exp, None)
        return None

    if kind == "side_swap":
        marker = "\x00"
        t = re.sub(r"\bleft\b", marker, text, flags=re.IGNORECASE)
        t = re.sub(r"\bright\b", "Left", t, flags=re.IGNORECASE).replace(marker, "Right")
        t = re.sub(r"\besquerda\b", marker, t, flags=re.IGNORECASE)
        t = re.sub(r"\bdireita\b", "esquerda", t, flags=re.IGNORECASE).replace(marker, "direita")
        return Injection(kind, t, None, None, exp, None)

    table = {"severity_change": SEVERITY_SWAPS, "location_change": LOCATION_SWAPS,
             "descriptor_swap": DESCRIPTOR_SWAPS, "density_change": DENSITY_SWAPS}.get(kind)
    if table is not None:
        if kind == "severity_change":
            m = re.search(r"(bi[\-\s]?rads?\s*[-:\(]?\s*)(0|1|2|3|4a|4b|4c|4|5|6)\b",
                          text, re.IGNORECASE)
            if not m:
                return None
            old = m.group(2).lower()
            new = SEVERITY_SWAPS.get(old)
            if not new:
                return None
            t = text[:m.start(2)] + new + text[m.end(2):]
            return Injection(kind, t, m.group(0), m.group(1) + new, exp,
                             (m.start(), m.end()))
        res = _swap_first(text, table)
        if res is None:
            return None
        t, src, dst = res
        return Injection(kind, t, src, dst, exp, None)

    if kind == "fabricated":
        sent = FABRICATED_SENTENCES[language]
        lines = text.splitlines(keepends=True)
        pos = rng.randrange(1, max(len(lines), 2))
        lines.insert(pos, sent + "\n")
        return Injection(kind, "".join(lines), None, sent, exp, None)

    return None


def build_benchmark(reports: dict[str, str], language: str = "en",
                    per_report: int = 1, seed: int = 20260819) -> list[dict]:
    """Gera a camada sintetica. Retorna casos com rotulo para metrica por tipo.

    Inclua SEMPRE casos NAO perturbados (kind='clean'): sem eles o baseline
    'sempre sem erro' nao pode ser medido, e sem esse baseline nenhum numero
    de acuracia tem interpretacao.
    """
    rng = random.Random(seed)
    kinds = list(PERTURBATIONS)
    out = []
    for exam_id, text in reports.items():
        out.append({"exam_id": exam_id, "kind": "clean", "text": text,
                    "expected_alert": None})
        picks = [k for k in rng.sample(kinds, len(kinds)) if applicable(text, k)][:per_report]
        for kind in picks:
            inj = inject(text, kind, language, rng)
            if inj is None:
                continue
            out.append({"exam_id": exam_id, "kind": inj.kind, "text": inj.text,
                        "expected_alert": inj.expected_alert,
                        "original_sentence": inj.original_sentence,
                        "modified_sentence": inj.modified_sentence})
    return out
