"""Seccionador -- FACA ISTO PRIMEIRO. Evita ~80% dos falsos alertas.

Problema real no seu corpus atual (INbreast, unico com laudo em texto livre):
os laudos descrevem TAMBEM ultrassom, microbiopsia, axila, marcacao
pre-operatoria e histologia. Um laudo que diz 'Axila negativa' nao e
auditavel contra uma mamografia. Sinalizar isso como omissao e falso
alerta POR CONSTRUCAO.

Este modulo tambem sabe separar uma secao 'CONTRAST ENHANCED SPECTRAL
MAMMOGRAPHY REVEALED:' (mamografia contrastada / CESM) do restante do
laudo -- capacidade generica, mantida para o caso de voce incorporar no
futuro um corpus com esse formato. NENHUM dos datasets atuais do projeto
(VinDr, CBIS-DDSM, INbreast) tem laudo com secao CESM, entao essa parte
fica ociosa por enquanto (o CDD-CESM, que tinha esse formato, foi
removido do escopo em 24/08/2026).
"""
from __future__ import annotations
import re
from dataclasses import dataclass

DM_HEADERS = [
    r"DIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED",
    r"MAMMOGRAPHY REVEALED",
    r"DIGITAL MAMMOGRAPHY",
]
CESM_HEADERS = [
    r"CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED",
    r"CONTRAST ENHANCED",
    r"\bCESM\b",
]
OPINION_HEADERS = [r"^OPINION\s*:", r"^IMPRESS(AO|ÃO|ION)\s*:", r"^CONCLUS(AO|ÃO|ION)\s*:"]

NON_MAMMO_PT = [
    "ecografia", "ecograf", "ultrassonograf", "ultra-sonograf", "ultrassom",
    "microbiopsia", "micro-biopsia", "biopsia", "biópsia", "mb-",
    "anatomopatolog", "histolog", "marcacao pre-operatoria", "marcação pré-operatória",
    "axila", "axilar", "puncao", "punção", "citolog",
]

# Cabecalhos de mama: ANCORADOS EM INICIO DE LINHA.
# Nao ancorar aqui parte sentencas em prosa no meio ("...da mama direita com 2 cm...")
# e destroi a extracao dos laudos portugueses do INbreast.
SIDE_HEADERS = {
    "L": [r"^\s*Left Breast\s*:", r"^\s*mama esquerda", r"^\s*Mama Esquerda"],
    "R": [r"^\s*Right Breast\s*:", r"^\s*mama direita", r"^\s*Mama Direita"],
}

# Mencoes de lateralidade EM PROSA: usadas por sentenca, nunca para cortar o texto.
SIDE_INLINE = {
    "L": [r"\bleft breast\b", r"\bmama esquerda\b", r"\besquerda\b", r"\besquerdo\b"],
    "R": [r"\bright breast\b", r"\bmama direita\b", r"\bdireita\b", r"\bdireito\b"],
}


def detect_side(snippet: str) -> str | None:
    """Lateralidade de uma sentenca. None se ambigua ou ausente."""
    hits = {
        side for side, pats in SIDE_INLINE.items()
        if any(re.search(p, snippet, re.IGNORECASE) for p in pats)
    }
    return hits.pop() if len(hits) == 1 else None


@dataclass
class Section:
    modality: str            # DM | CESM | UNKNOWN
    part: str                # FINDINGS | OPINION
    laterality: str | None   # L | R | None
    text: str
    span: tuple[int, int]


def _find(patterns: list[str], text: str) -> int:
    best = -1
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m and (best < 0 or m.start() < best):
            best = m.start()
    return best


def split_by_modality(text: str) -> dict[str, tuple[int, int]]:
    """Retorna {'DM': (ini, fim), 'CESM': (ini, fim)} em offsets de caractere."""
    dm_i = _find(DM_HEADERS, text)
    ce_i = _find(CESM_HEADERS, text)
    out: dict[str, tuple[int, int]] = {}
    if dm_i >= 0:
        out["DM"] = (dm_i, ce_i if ce_i > dm_i else len(text))
    if ce_i >= 0:
        out["CESM"] = (ce_i, len(text))
    if not out:
        out["UNKNOWN"] = (0, len(text))
    return out


def split_by_side(text: str, offset: int = 0) -> list[tuple[str | None, str, tuple[int, int]]]:
    """Divide um bloco em sub-blocos por mama. Preserva os offsets absolutos."""
    marks: list[tuple[int, str]] = []
    for side, pats in SIDE_HEADERS.items():
        for p in pats:
            for m in re.finditer(p, text, re.IGNORECASE | re.MULTILINE):
                marks.append((m.start(), side))
    marks.sort()
    if not marks:
        return [(None, text, (offset, offset + len(text)))]
    out = []
    for i, (pos, side) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((side, text[pos:end], (offset + pos, offset + end)))
    return out


def sectionize(text: str, language: str = "en") -> list[Section]:
    sections: list[Section] = []
    for modality, (start, end) in split_by_modality(text).items():
        block = text[start:end]
        op_i = _find(OPINION_HEADERS, block)
        parts = (
            [("FINDINGS", block[:op_i], start), ("OPINION", block[op_i:], start + op_i)]
            if op_i > 0
            else [("FINDINGS", block, start)]
        )
        for part, ptext, poff in parts:
            for side, stext, span in split_by_side(ptext, poff):
                sections.append(Section(modality, part, side, stext, span))
    return sections


def auditable_sections(text: str, language: str = "en",
                       target_modality: str = "DM") -> list[Section]:
    """So o que pode ser confrontado com uma imagem de mamografia convencional.

    Descarta: (a) a secao CESM; (b) sentencas sobre ultrassom/biopsia/axila.
    """
    keep = []
    for sec in sectionize(text, language):
        if sec.modality not in (target_modality, "UNKNOWN"):
            continue
        low = sec.text.lower()
        lines, cursor = [], sec.span[0]
        for line in sec.text.splitlines(keepends=True):
            if not any(k in line.lower() for k in NON_MAMMO_PT):
                lines.append(line)
            cursor += len(line)
        filtered = "".join(lines).strip()
        if filtered:
            keep.append(Section(sec.modality, sec.part, sec.laterality, filtered, sec.span))
    return keep
