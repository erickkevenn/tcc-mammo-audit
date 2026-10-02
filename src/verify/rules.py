"""Regras V1 a V4: a categoria BI-RADS do laudo combina com o exame?

Deterministico de proposito: cada alerta aponta a regra, o limiar e a evidencia.
O sistema NAO diz qual categoria e a correta; diz que a do laudo parece nao
combinar com o que a imagem mostra (V1, V2, V4) ou com o que o proprio laudo
descreve (V3).

  V1  achado suspeito na imagem (massa ou calcificacao acima do limiar) e
      categoria 1 ou 2 no laudo
  V2  categoria 4 ou 5 no laudo e nenhum achado na imagem acima do limiar
      (prioridade menor: pode ser achado que os detectores nao cobrem)
  V3  so o texto: achado descrito como suspeito com categoria 1 ou 2; categoria
      4 ou 5 com todos os achados descritos como benignos; categoria 1 com
      achado afirmado
  V4  categoria do laudo e a estimada pelo classificador em lados opostos da
      fronteira de biopsia (1 a 3 contra 4 e 5), com limiar na P(4)+P(5)

Categorias 0 (incompleta), 6 (malignidade comprovada) ou ausentes nao sao
verificadas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..report.schema import birads_to_num

PRIORITY = {"V1": 1, "V3": 1, "V4": 2, "V2": 3}     # 1 = mais importante


@dataclass
class Thresholds:
    mass: float          # confianca minima do detector de massas (por mama)
    calc: float          # confianca minima do detector de microcalcificacoes (por mama)
    cls_ge4: float       # P(4)+P(5) a partir da qual o classificador poe a mama no lado da biopsia


@dataclass
class BreastImage:
    mass_score: float = 0.0          # maior confianca de massa entre as vistas da mama
    calc_score: float = 0.0          # maior confianca de calcificacao entre as vistas
    p_ge4: float | None = None       # P(4)+P(5) do classificador (media das vistas)
    mass_box: list | None = None     # evidencia: caixa e vista da maior deteccao
    calc_box: list | None = None


@dataclass
class ReportFinding:
    category: str                    # mass | suspicious_calcification | ...
    affirmed: bool
    suspicion: str | None = None     # benign | suspicious | None
    text: str | None = None          # trecho do laudo (evidencia)


@dataclass
class Alert:
    rule: str
    priority: int
    message: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def biopsy_side(category: str | None) -> int | None:
    """1 = categorias 4 e 5 (biopsia), 0 = 1 a 3; None = nao verificavel (0, 6, ausente)."""
    n = birads_to_num(category)
    if n is None or n < 1 or n >= 6:
        return None
    return int(n >= 4)


def verify_breast(category: str | None, image: BreastImage | None, findings: list[ReportFinding] | None,
                  thr: Thresholds) -> list[Alert]:
    """Alertas para UMA mama. image=None desliga V1, V2 e V4; findings=None desliga V3."""
    side = biopsy_side(category)
    if side is None:
        return []
    n = birads_to_num(category)
    alerts: list[Alert] = []

    if image is not None:
        mass_hit = image.mass_score >= thr.mass
        calc_hit = image.calc_score >= thr.calc
        if n <= 2 and (mass_hit or calc_hit):
            what = [w for w, h in (("massa", mass_hit), ("microcalcificacao", calc_hit)) if h]
            alerts.append(Alert("V1", PRIORITY["V1"],
                                f"achado suspeito na imagem ({', '.join(what)}) com categoria {category}",
                                {"mass_score": image.mass_score, "calc_score": image.calc_score,
                                 "mass_box": image.mass_box, "calc_box": image.calc_box}))
        if side == 1 and not mass_hit and not calc_hit:
            alerts.append(Alert("V2", PRIORITY["V2"],
                                f"categoria {category} sem achado na imagem acima do limiar",
                                {"mass_score": image.mass_score, "calc_score": image.calc_score}))
        if image.p_ge4 is not None:
            est = int(image.p_ge4 >= thr.cls_ge4)
            if est != side:
                alerts.append(Alert("V4", PRIORITY["V4"],
                                    f"categoria {category} e classificador do outro lado da fronteira de biopsia",
                                    {"p_ge4": image.p_ge4}))

    if findings is not None:
        aff = [f for f in findings if f.affirmed and f.category not in ("no_finding", "postoperative")]
        sus = [f for f in aff if f.suspicion == "suspicious"]
        if n <= 2 and sus:
            alerts.append(Alert("V3", PRIORITY["V3"], f"laudo descreve achado suspeito com categoria {category}",
                                {"trechos": [f.text for f in sus]}))
        elif side == 1 and aff and all(f.suspicion == "benign" for f in aff):
            alerts.append(Alert("V3", PRIORITY["V3"],
                                f"laudo com categoria {category} descreve so achados benignos",
                                {"trechos": [f.text for f in aff]}))
        elif n == 1 and aff:
            alerts.append(Alert("V3", PRIORITY["V3"], "laudo com categoria 1 (negativo) afirma achado",
                                {"trechos": [f.text for f in aff]}))
    return sorted(alerts, key=lambda a: a.priority)
