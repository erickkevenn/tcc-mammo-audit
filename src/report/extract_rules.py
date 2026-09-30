"""E0 -- extrator por regras (lexico + negacao estilo ConText).

E o BASELINE OBRIGATORIO. Espere precisao ruim na negacao: no estudo de
referencia, medspaCy teve F1 0,492 (precisao 0,356) contra 0,777 do CAN-BERT
em 984 laudos. Reporte isso -- e a justificativa para o E1 (LLM local).

A categoria BI-RADS ESCRITA no laudo deve ser extraida por REGEX, nunca por LLM:
GPT-4 teve concordancia AC1 de apenas 0,57 (ingles) / 0,50 (italiano) /
0,49 (holandes) contra 0,91 humano-humano, com mudanca negativa de conduta em
10,6% dos casos vs 1,5% entre humanos (Radiology, 2024).
"""
from __future__ import annotations
import re
import unicodedata

from ..config import lexicon
from .schema import (Descriptors, Evidence, Finding, FindingRecord, Region, Status)
from .sectionizer import auditable_sections, detect_side

BIRADS_RE = re.compile(
    r"\bbi[\-\s]?rads?\s*[-:\(]?\s*(?P<cat>0|1|2|3|4a|4b|4c|4|5|6)\b", re.IGNORECASE
)
SIZE_RE = re.compile(r"(?P<v>\d+(?:[.,]\d+)?)\s*(?P<u>mm|cm)\b", re.IGNORECASE)


def norm(text: str) -> str:
    """Minusculas sem acento -- casa 'nódulo' com 'nodulo' sem duplicar o lexico."""
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


class RuleExtractor:
    def __init__(self, language: str = "en"):
        self.lex = lexicon()
        self.lang = language
        self.neg = [norm(x) for x in self.lex["negation"].get(language, [])]
        self.terms = [norm(x) for x in self.lex["negation_terminators"].get(language, [])]

    # --------------------------------------------------------------- utilidades
    QUADRANT_KEYS = {"UOQ", "UIQ", "LOQ", "LIQ", "retroareolar", "axillary"}

    def _match_group(self, group: str, text: str, allowed: set[str] | None = None) -> str | None:
        """Retorna a chave canonica cujo sinonimo mais longo aparece no texto."""
        best, best_len = None, 0
        for key, langs in self.lex.get(group, {}).items():
            if allowed is not None and key not in allowed:
                continue
            if not isinstance(langs, dict):
                continue
            for syn in langs.get(self.lang, []) + langs.get("en", []):
                s = norm(str(syn))
                if s and s in text and len(s) > best_len:
                    best, best_len = key, len(s)
        return best

    def _negated(self, sentence_norm: str, cue_pos: int) -> bool:
        """Negacao pre-posta com terminador de escopo (ConText simplificado)."""
        window = sentence_norm[:cue_pos]
        for n in self.neg:
            idx = window.rfind(n)
            if idx < 0:
                continue
            between = window[idx + len(n):]
            if any(t in between for t in self.terms):
                continue          # o escopo da negacao terminou antes do achado
            if len(between) <= 60:
                return True
        return False

    # ------------------------------------------------------------------ publico
    def extract(self, text: str, exam_id: str, laterality: str,
                target_modality: str = "DM") -> FindingRecord:
        rec = FindingRecord(
            exam_id=exam_id, laterality=laterality, source="report",
            modality_section=target_modality, language=self.lang, extractor="rules",
        )
        sections = [
            s for s in auditable_sections(text, self.lang, target_modality)
            if s.laterality in (laterality, None)
        ]
        seen: set[tuple] = set()
        for sec in sections:
            for sentence in re.split(r"(?<=[.\n])\s*", sec.text):
                if not sentence.strip():
                    continue
                # Lateralidade por sentenca tem prioridade sobre a da secao:
                # laudos em prosa (INbreast) nao usam cabecalho por mama.
                sent_side = detect_side(sentence)
                if sent_side is not None and sent_side != laterality:
                    continue
                sn = norm(sentence)
                offset = sec.span[0] + sec.text.find(sentence)
                for cat, langs in self.lex["category"].items():
                    hit_pos, hit_len = -1, 0
                    for syn in langs.get(self.lang, []) + langs.get("en", []):
                        s = norm(str(syn))
                        p = sn.find(s)
                        if s and p >= 0 and len(s) > hit_len:
                            hit_pos, hit_len = p, len(s)
                    if hit_pos < 0:
                        continue
                    status = Status.NEGATED if self._negated(sn, hit_pos) else Status.AFFIRMED
                    if cat == "no_finding":
                        status = Status.AFFIRMED
                    # Tamanho so faz sentido para achados mensuraveis. Sem este
                    # filtro, '2 cm' contamina a calcificacao negada da mesma sentenca.
                    size = None
                    if cat in ("mass", "focal_asymmetry", "asymmetry",
                               "architectural_distortion", "suspicious_lymph_node"):
                        ms = SIZE_RE.search(sentence)
                        if ms:
                            v = float(ms.group("v").replace(",", "."))
                            size = v * 10 if ms.group("u").lower() == "cm" else v
                    bm = BIRADS_RE.search(sentence)
                    key = (cat, status, offset)
                    if key in seen:
                        continue
                    seen.add(key)
                    rec.findings.append(
                        Finding(
                            category=cat, status=status,
                            region=Region(
                                quadrant=self._match_group("region", sn, self.QUADRANT_KEYS),
                                vertical="upper" if "upper" in sn or "superior" in sn else
                                         ("lower" if "lower" in sn or "inferior" in sn else None),
                                horizontal="outer" if "outer" in sn or "externo" in sn else
                                           ("inner" if "inner" in sn or "interno" in sn else None),
                            ),
                            descriptors=Descriptors(
                                mass_shape=self._match_group("mass_shape", sn),
                                mass_margin=self._match_group("mass_margin", sn),
                                mass_density=self._match_group("mass_density", sn),
                                calc_morphology=self._match_group("calc_morphology", sn),
                                calc_distribution=self._match_group("calc_distribution", sn),
                                size_mm=size,
                            ),
                            birads=bm.group("cat").lower() if bm else None,
                            evidence=Evidence(
                                sentence_span=[offset, offset + len(sentence)],
                                sentence_text=sentence.strip(),
                            ),
                        )
                    )
        # 'no_finding' e mutuamente exclusivo com achado afirmado: 'sem
        # microcalcificacoes' numa mama que TEM nodulo nao e exame normal.
        has_positive = any(
            f.status == Status.AFFIRMED and f.category not in ("no_finding", "postoperative")
            for f in rec.findings
        )
        if has_positive:
            rec.findings = [f for f in rec.findings if f.category != "no_finding"]

        # BI-RADS e densidade por mama: preferir a secao OPINION quando existir.
        op = [s for s in sections if s.part == "OPINION"] or sections
        joined = " ".join(s.text for s in op)
        bm = BIRADS_RE.search(joined)
        if bm:
            rec.breast_birads = bm.group("cat").lower()
        rec.breast_density = self._match_group("density_acr", norm(text))
        return rec


if __name__ == "__main__":
    demo = (
        "PATIENT NO.1\nDIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED:\n"
        "ACR C: Heterogeneously dense breasts.\nLeft Breast:\n"
        "Upper architectural distortion is seen.\nNo suspicious microcalcifications.\n"
        "OPINION:\nLeft Breast:\nUpper architectural distortion (BIRADS 4).\n"
        "CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:\n"
        "Left Breast:\nUpper heterogeneous non mass enhancement (BIRADS 4).\n"
    )
    r = RuleExtractor("en").extract(demo, "P1", "L")
    print(r.model_dump_json(indent=2, exclude_none=True))
