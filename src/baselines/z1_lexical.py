"""Z1 -- baseline lexical: procura a palavra do achado no laudo, SEM olhar a imagem.

Mede quanto do problema e resolvido por casamento de palavra-chave. Se o seu
sistema completo nao superar Z1 com folga, o modulo de visao nao esta
contribuindo -- e melhor descobrir isso em novembro do que na banca.
"""
from __future__ import annotations
from ..report.extract_rules import norm
from ..config import lexicon

LEX = lexicon()


def has_category(text: str, category: str, language: str = "en") -> bool:
    t = norm(text)
    langs = LEX["category"].get(category, {})
    syns = langs.get(language, []) + langs.get("en", [])
    return any(norm(str(s)) in t for s in syns)


def predict(text: str, expected_categories: list[str], language: str = "en") -> list[str]:
    """Retorna as categorias esperadas que NAO aparecem no texto (proxy de A1)."""
    return [c for c in expected_categories if not has_category(text, c, language)]
