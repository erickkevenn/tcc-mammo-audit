"""M4 -- o auditor. DETERMINISTICO de proposito.

Por que nao um modelo: rastreabilidade. Cada alerta tem de ser explicavel por
uma regra citavel e um limiar declarado. E o unico componente que voce consegue
depurar em dezembro sem retreinar nada.

Taxonomia (adaptada de CheXprompt / GREEN / CorBenchX ao BI-RADS):
  A1 Omissao de achado suspeito         (imagem contem, laudo nao afirma)
  A2 Achado afirmado sem sustentacao    (laudo afirma, imagem nao ancora)
  A3 Discordancia de lateralidade
  A4 Discordancia de localizacao
  A5 Discordancia de descritor          <- diferencial do TCC
  A6 Inconsistencia de categoria BI-RADS
  A7 Discordancia de densidade ACR
  A8 Inconsistencia interna do laudo    <- nao depende do modulo de visao (plano B)

Severidade: escala de 5 niveis do ReFiSco
  1 sem erro | 2 nao acionavel | 3 acionavel nao urgente | 4 urgente | 5 emergente
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict

from ..config import load
from ..report.schema import (FindingRecord, Status, birads_to_num,
                             crosses_action_boundary)

DESCRIPTOR_FIELDS = ["mass_shape", "mass_margin", "mass_density",
                     "calc_morphology", "calc_distribution"]


@dataclass
class Alert:
    code: str
    severity: int
    message: str
    laterality: str
    exam_id: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Auditor:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load("audit_rules.yaml")
        self.tau = float(self.cfg["tau_detect"])
        self.tol = self.cfg["tolerances"]
        self.adj = self.cfg.get("region_adjacency", {})
        self.sev = self.cfg["severity"]
        self.enabled = set(self.cfg.get("enabled_rules", []))

    # -------------------------------------------------------------- utilidades
    def _region_compatible(self, a, b) -> bool:
        """'unspecified'/None casa com tudo. Quadrante casa com seus adjacentes."""
        if a is None or b is None:
            return True
        if a == b:
            return True
        return b in self.adj.get(a, []) or a in self.adj.get(b, [])

    def _same_finding(self, img_f, rep_f) -> bool:
        if img_f.category != rep_f.category:
            if {img_f.category, rep_f.category} <= {"asymmetry", "focal_asymmetry",
                                                    "global_asymmetry"}:
                pass                      # sinonimos aceitos
            else:
                return False
        return self._region_compatible(img_f.region.quadrant, rep_f.region.quadrant)

    def _severity(self, code: str, **flags) -> int:
        spec = self.sev.get(code, {"default": 3})
        for key, val in flags.items():
            if val and key in spec:
                return int(spec[key])
        return int(spec.get("default", 3))

    # ------------------------------------------------------------------ regras
    def audit(self, image: FindingRecord, report: FindingRecord) -> list[Alert]:
        assert image.laterality == report.laterality, "audite uma mama por vez"
        alerts: list[Alert] = []
        exam, side = image.exam_id, image.laterality

        img_conf = [f for f in image.findings
                    if (f.confidence or 1.0) >= self.tau and f.category != "no_finding"]
        rep_aff = [f for f in report.findings
                   if f.status == Status.AFFIRMED and f.category != "no_finding"]
        rep_neg = [f for f in report.findings if f.status == Status.NEGATED]

        # A1 -- omissao de achado suspeito
        if "A1" in self.enabled:
            for f in img_conf:
                if any(self._same_finding(f, r) for r in rep_aff):
                    continue
                bn = birads_to_num(f.birads) or 0
                explicitly_negated = any(n.category == f.category for n in rep_neg)
                alerts.append(Alert(
                    code="A1",
                    severity=self._severity("A1", birads5=bn >= 5, birads4=4 <= bn < 5),
                    message=(
                        f"Achado '{f.category}' detectado na mama {side} "
                        f"(conf={f.confidence:.2f}, BI-RADS visual={f.birads}) "
                        + ("EXPLICITAMENTE NEGADO no laudo." if explicitly_negated
                           else "nao mencionado no laudo.")
                    ),
                    laterality=side, exam_id=exam,
                    evidence={"bbox": f.evidence.bbox, "image_id": f.evidence.image_id,
                              "explicitly_negated": explicitly_negated},
                ))

        # A2 -- achado afirmado sem sustentacao na imagem
        if "A2" in self.enabled:
            for r in rep_aff:
                if any(self._same_finding(f, r) for f in img_conf):
                    continue
                alerts.append(Alert(
                    code="A2", severity=self._severity("A2"),
                    message=(f"Laudo afirma '{r.category}' na mama {side}, "
                             f"sem regiao correspondente na imagem."),
                    laterality=side, exam_id=exam,
                    evidence={"sentence": r.evidence.sentence_text,
                              "span": r.evidence.sentence_span},
                ))

        # A4/A5 -- discordancia de localizacao e de descritor
        for r in rep_aff:
            for f in img_conf:
                if r.category != f.category:
                    continue
                if "A4" in self.enabled and not self._region_compatible(
                        f.region.quadrant, r.region.quadrant):
                    alerts.append(Alert(
                        code="A4", severity=self._severity("A4"),
                        message=(f"'{r.category}': laudo diz {r.region.quadrant}, "
                                 f"imagem indica {f.region.quadrant}."),
                        laterality=side, exam_id=exam,
                        evidence={"sentence": r.evidence.sentence_text,
                                  "bbox": f.evidence.bbox},
                    ))
                if "A5" in self.enabled:
                    for fld in DESCRIPTOR_FIELDS:
                        rv = getattr(r.descriptors, fld)
                        fv = getattr(f.descriptors, fld)
                        if rv and fv and rv != fv:
                            alerts.append(Alert(
                                code="A5", severity=self._severity("A5"),
                                message=(f"Descritor '{fld}': laudo='{rv}' vs "
                                         f"predicao='{fv}' ({r.category})."),
                                laterality=side, exam_id=exam,
                                evidence={"sentence": r.evidence.sentence_text,
                                          "field": fld},
                            ))
                # tamanho
                rs, fs = r.descriptors.size_mm, f.descriptors.size_mm
                if rs and fs:
                    tol = max(self.tol["size_abs_mm"], self.tol["size_rel"] * rs)
                    if abs(rs - fs) > tol:
                        alerts.append(Alert(
                            code="A5", severity=self._severity("A5"),
                            message=f"Tamanho: laudo={rs:.0f} mm vs predicao={fs:.0f} mm.",
                            laterality=side, exam_id=exam,
                            evidence={"sentence": r.evidence.sentence_text, "field": "size_mm"},
                        ))

        # A6 -- BI-RADS por mama
        if "A6" in self.enabled and image.breast_birads and report.breast_birads:
            ni, nr = birads_to_num(image.breast_birads), birads_to_num(report.breast_birads)
            crosses = crosses_action_boundary(image.breast_birads, report.breast_birads)
            if ni is not None and nr is not None and (
                    abs(ni - nr) >= self.tol["birads_levels"]
                    or (crosses and self.tol.get("birads_action_boundary"))):
                alerts.append(Alert(
                    code="A6",
                    severity=self._severity("A6", crosses_action_boundary=crosses),
                    message=(f"BI-RADS: laudo={report.breast_birads} vs "
                             f"predicao={image.breast_birads}"
                             + (" (CRUZA a fronteira de conduta <=3 / >=4)" if crosses else "")),
                    laterality=side, exam_id=exam,
                    evidence={"crosses_action_boundary": crosses},
                ))

        # A7 -- densidade ACR
        if "A7" in self.enabled and image.breast_density and report.breast_density:
            order = {"A": 1, "AB": 1.5, "B": 2, "C": 3, "D": 4}
            di, dr = order.get(image.breast_density), order.get(report.breast_density)
            if di and dr and abs(di - dr) >= self.tol["density_levels"]:
                alerts.append(Alert(
                    code="A7", severity=self._severity("A7"),
                    message=(f"Densidade ACR: laudo={report.breast_density} vs "
                             f"predicao={image.breast_density}"),
                    laterality=side, exam_id=exam, evidence={},
                ))

        return alerts

    # A8 -- nao depende do modulo de visao. Implemente na semana 12: e barato,
    # tem alto valor clinico e funciona como plano B se o detector decepcionar.
    def audit_report_internal(self, report: FindingRecord,
                              opinion: FindingRecord | None = None) -> list[Alert]:
        alerts: list[Alert] = []
        if "A8" not in self.enabled:
            return alerts
        suspicious = {"spiculated", "irregular", "fine_pleomorphic", "fine_linear"}
        nb = birads_to_num(report.breast_birads)
        for f in report.findings:
            if f.status != Status.AFFIRMED:
                continue
            descr = {getattr(f.descriptors, k) for k in DESCRIPTOR_FIELDS}
            if descr & suspicious and nb is not None and nb <= 2:
                alerts.append(Alert(
                    code="A8", severity=self._severity("A8"),
                    message=(f"Descritor suspeito ({descr & suspicious}) com "
                             f"BI-RADS {report.breast_birads} no proprio laudo."),
                    laterality=report.laterality, exam_id=report.exam_id,
                    evidence={"sentence": f.evidence.sentence_text},
                ))
        if opinion is not None:
            cats_f = {f.category for f in report.findings if f.status == Status.AFFIRMED}
            cats_o = {f.category for f in opinion.findings if f.status == Status.AFFIRMED}
            for missing in cats_f - cats_o:
                alerts.append(Alert(
                    code="A8", severity=self._severity("A8"),
                    message=f"'{missing}' descrito em Findings e ausente na Opinion.",
                    laterality=report.laterality, exam_id=report.exam_id, evidence={},
                ))
        return alerts
