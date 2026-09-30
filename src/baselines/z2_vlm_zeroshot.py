"""Z2 -- VLM zero-shot (imagem + laudo -> ha erro?). Sua comparacao com o SOTA.

A barra publicada e 50,6% de acuracia de deteccao (o4-mini no CorBenchX, tarefa
formalmente identica em radiografia de torax). Em mamografia especificamente,
o MammoVQA (Nature Communications 2025, 12 VLMs) concluiu que a maioria dos
LVLMs pre-treinados ficou 'estatisticamente equivalente a palpite aleatorio'.
MedGemma nao tem mamografia nem no treino nem na avaliacao.

Candidatos que cabem em 8 GB em 4-bit: Qwen2.5-VL-7B, MedGemma-4B-it, LLaVA-Med-7B
(via Ollama, como faz o MammoWise). Implemente na semana 14.
"""
from __future__ import annotations

PROMPT = """Voce recebe uma mamografia e o laudo correspondente.
Pergunta 1: existe algum erro no laudo em relacao a imagem? (sim/nao)
Pergunta 2: se sim, qual o tipo? (omissao | achado inexistente | lateralidade |
localizacao | descritor | categoria BI-RADS | densidade)
Pergunta 3: qual a severidade? (1 a 5)
Responda em JSON: {"has_error": bool, "type": str|null, "severity": int}
"""


def build_prompt(report_text: str) -> str:
    return PROMPT + "\n\nLaudo:\n\"\"\"\n" + report_text + "\n\"\"\"\n"


class VLMBaseline:
    def __init__(self, model_id: str = "qwen2.5-vl:7b", backend: str = "ollama"):
        self.model_id = model_id
        self.backend = backend
        raise NotImplementedError(
            "Semana 14: conecte via Ollama (local) ou transformers 4-bit. "
            "Registre a versao exata do modelo -- e item do CLAIM 2024."
        )
