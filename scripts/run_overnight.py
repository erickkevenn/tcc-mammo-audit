"""Roda em sequencia o que falta da S15 e grava um log por etapa (para deixar rodando).

Cada etapa roda mesmo se a anterior falhar. No fim, artifacts/overnight/resumo.txt
diz o que deu certo. Nada aqui muda modelo, limiar ou par congelado.

  1. testes automaticos
  2. IC 95% do classificador no teste (predicoes ja salvas)
  3. teste externo do detector de massas no CBIS-DDSM (uma vez, modelo congelado)
  4. analise de erro: massas, microcalcificacoes, verificacao e alertas da V3

Uso:  python scripts/run_overnight.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STEPS = [
    ("testes", [sys.executable, "-m", "pytest", "-q", "tests"]),
    ("classificador_ic", [sys.executable, "scripts/cls_test_ci.py"]),
    ("cbis_massas", [sys.executable, "scripts/eval_cbis_mass.py", "--confirm-test"]),
    ("analise_erro", [sys.executable, "scripts/error_analysis.py"]),
]


def main() -> None:
    out = Path("artifacts/overnight")
    out.mkdir(parents=True, exist_ok=True)
    resumo = []
    for i, (name, cmd) in enumerate(STEPS, 1):
        log = out / f"{i}_{name}.log"
        t0 = time.time()
        print(f"[{datetime.now():%H:%M}] {i}/{len(STEPS)} {name} -> {log}", flush=True)
        with log.open("w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        status = "ok" if rc == 0 else f"ERRO (codigo {rc})"
        line = f"{i}. {name}: {status} em {(time.time() - t0) / 60:.1f} min | log: {log}"
        print("   " + line, flush=True)
        resumo.append(line)
    (out / "resumo.txt").write_text("\n".join(resumo) + "\n", encoding="utf-8")
    print("\n" + "\n".join(resumo))
    print(f"\nresumo -> {out / 'resumo.txt'}")


if __name__ == "__main__":
    main()
