"""E1 -- extrator com LLM local (4-bit) e saida JSON RESTRITA.

Por que vale a pena: LLMs de pesos abertos em zero-shot empatam com o GPT-4o
e destroem regras na rotulagem de achados -- F1 macro 0,926 (Mistral-Large)
vs 0,924 (GPT-4o) vs 0,731 (CheXpert de regras). Em aleme, 0,916 vs 0,748.
E funciona pequeno: Qwen1.5-0.5B com instruction tuning atingiu F1 0,9014 em
14 patologias, superando o CheXpert (0,8864, p=0,01).

Restricoes de honestidade do TCC:
  * NAO peca ao LLM a categoria BI-RADS final (AC1 ~0,5). Extraia por regex.
  * Se usar LLM para INJETAR erro e LLM para DETECTAR, use modelos e prompts
    DIFERENTES e declare isso -- caso contrario ha vazamento de vies de geracao.

Este arquivo e um esqueleto deliberado: escolha o runtime na semana 11.
Opcoes que cabem em 8 GB: llama.cpp com gramatica GBNF; vLLM com guided_json;
Outlines; ou transformers + bitsandbytes 4-bit + validacao pydantic com retry.
"""
from __future__ import annotations
import json

from .schema import FindingRecord

SYSTEM_PROMPT = """Voce extrai achados de laudos de mamografia para AUDITORIA.
Responda APENAS com JSON valido no esquema fornecido. Regras:
- Nao invente achados. Se o laudo nao menciona, omita.
- Distinga NEGADO ("sem microcalcificacoes") de AUSENTE (nao mencionado):
  use status="negated" para o primeiro e nao crie o achado para o segundo.
- Copie o trecho exato do laudo em evidence.sentence_text.
- Nao preencha o campo birads: ele e extraido separadamente.
"""

USER_TEMPLATE = """Esquema JSON:
{schema}

Laudo (mama {laterality}, secao {modality}):
\"\"\"
{text}
\"\"\"

JSON:"""


def build_prompt(text: str, laterality: str, modality: str = "DM") -> list[dict]:
    schema = FindingRecord.model_json_schema()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                schema=json.dumps(schema, ensure_ascii=False)[:4000],
                laterality=laterality, modality=modality, text=text,
            ),
        },
    ]


def parse_response(raw: str, exam_id: str, laterality: str) -> FindingRecord:
    """Valida com pydantic. Em caso de falha, chame o modelo de novo com o erro
    no prompt (retry guiado por erro e mais eficaz que temperatura menor)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("nenhum objeto JSON na resposta")
    data = json.loads(raw[start:end + 1])
    data.setdefault("exam_id", exam_id)
    data.setdefault("laterality", laterality)
    data["source"] = "report"
    data["extractor"] = "llm"
    return FindingRecord.model_validate(data)


class LocalLLMExtractor:
    """Preencha na semana 11. Mantenha a interface identica ao RuleExtractor
    para que a comparacao E0 x E1 seja um for loop, nao um refactor."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", quant: str = "4bit"):
        self.model_id = model_id
        self.quant = quant
        raise NotImplementedError(
            "Escolha o runtime (llama.cpp+GBNF | vLLM guided_json | transformers 4-bit) "
            "e implemente .extract(). O prompt e o parser ja estao prontos acima."
        )

    def extract(self, text: str, exam_id: str, laterality: str,
                target_modality: str = "DM") -> FindingRecord:
        raise NotImplementedError
