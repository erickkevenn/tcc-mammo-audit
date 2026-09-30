# Auditoria automática de laudos de mamografia

Sistema de verificação multimodal: cruza a **imagem** de mamografia com o **laudo já emitido**
e sinaliza (1) **omissões** de achado suspeito e (2) **discordâncias** de descritor BI-RADS.
Audita, não substitui o radiologista.

TCC — Erick Alves, Edge/UFAL, 2026. Plano completo: `PLANO_TCC_auditoria_mamografia.md`.

---

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Depois **ajuste apenas `configs/paths.yaml`** (ou exporte `TCC_DATA_ROOT`) para apontar
para a pasta dos datasets. Nenhum outro arquivo contém caminho absoluto.

```bash
export TCC_DATA_ROOT="D:/Erick/Dataset"     # Windows PowerShell: $env:TCC_DATA_ROOT="D:/Erick/Dataset"
pytest                                       # 61 testes, roda sem GPU e sem dados
```

## Ordem de execução

```bash
# 1. Triagem de metadados DICOM (sem ler pixels). Descobre as imagens FOR PROCESSING.
python -m src.data.build_manifest --dataset vindr --limit 200
python -m src.data.build_manifest --dataset vindr            # tudo (1,5-4 h)

# 2. Sanidade dos parsers -- compare com os numeros do docstring de cada modulo
python -m src.data.vindr
python -m src.data.cbis

# 3. Fatia vertical SEM GPU: laudo real (INbreast, PT-BR) -> extracao -> auditoria -> HTML
python scripts/demo_end_to_end.py --side R

# 4. Conversao para YOLO, por braco da ablacao de pre-processamento
python -m src.preprocess.to_yolo --arm B0 --limit 300      # amostra
python -m src.preprocess.to_yolo --arm B0                  # completo

# 5. Treino do detector (perfil <= 8 GB de VRAM)
python -m src.vision.train_detector --data D:/Erick/tcc_cache/yolo_B0/data.yaml
```

## Mapa dos módulos

| Caminho | O que faz |
|---|---|
| `src/data/build_manifest.py` | Etapa 0: metadados DICOM → `manifest.parquet` (item 13 do CLAIM 2024) |
| `src/data/vindr.py` · `cbis.py` · `inbreast.py` | parsers, cada um com as armadilhas do dataset documentadas no docstring |
| `src/data/splits.py` | split **por paciente** determinístico + teste anti-vazamento |
| `src/preprocess/dicom_io.py` | Modality LUT → VOI/janelamento → MONOCHROME1 **na ordem correta** |
| `src/preprocess/breast_roi.py` | recorte da mama (+4,4 AUC), flip canônico, resize+pad, mapeamento de bbox |
| `src/preprocess/to_yolo.py` | dataset YOLO por braço de ablação `B0`–`B4` |
| `src/vision/train_detector.py` | YOLO, 1024×640, batch 2 + acumulação 8, AMP |
| `src/vision/train_breast_classifier.py` | BI-RADS com **perda ordinal EMD** + densidade |
| `src/vision/tiles.py` · `aggregate.py` | caminho de calcificações; agregação imagem→mama→exame |
| `src/report/schema.py` | `FindingRecord` — o contrato entre imagem e texto |
| `src/report/sectionizer.py` | descarta menções a US/biópsia/axila (INbreast); suporta seção CESM de forma genérica, não usada nos datasets atuais |
| `src/report/extract_rules.py` | `E0`: léxico + negação (baseline obrigatório) |
| `src/report/extract_llm.py` | `E1`: LLM local 4-bit com JSON restrito (esqueleto) |
| `src/audit/rules.py` | auditor determinístico `A1`–`A8` + severidade 1–5 |
| `src/audit/matcher.py` | bbox → quadrante BI-RADS |
| `src/audit/report_html.py` | alerta com evidência visual |
| `src/eval/froc.py` | FROC: sensibilidade @ FP/imagem (métrica primária) |
| `src/eval/bootstrap.py` | IC 95% reamostrando **exames** (BCa) |
| `src/eval/stats.py` | DeLong, McNemar exato, kappa quadrático/Cohen/Fleiss |
| `src/eval/error_injection.py` | 7 perturbações **condicionadas ao contexto** |
| `src/baselines/` | `Z0` prior · `Z1` lexical · `Z2` VLM zero-shot |

## Cinco decisões de projeto que não devem ser revertidas

1. **Laudo real, nunca gerado a partir do rótulo.** Gerar o laudo da mesma anotação que
   serve de padrão de referência torna a auditoria circular e trivial.
2. **Split por paciente.** 4 vistas por exame e a mesma lesão em CC e MLO: split por
   imagem garante vazamento. Use o split oficial do VinDr (4.000/1.000).
3. **CLAHE é braço de ablação, não premissa.** Nenhuma solução vencedora do RSNA 2023
   usa CLAHE; o ganho publicado isolado é de +0,002 a +0,006 AUC. O ganho real está no
   recorte da mama e na resolução.
4. **`NEGATED` ≠ `ABSENT`.** "sem microcalcificações" e "não mencionou" são estados
   diferentes e geram alertas diferentes.
5. **O auditor é determinístico.** Cada alerta tem regra citável, limiar declarado e
   evidência rastreável. É o que diferencia auditoria de caixa-preta.

## Escopo declarado

O VinDr-Mammo só anota achados com BI-RADS > 2 (verificado: `finding_birads` ∈ {3, 4, 5}).
Portanto "omissão de achado" significa **omissão de achado suspeito (BI-RADS ≥ 3)**.
Não é limitação acidental — é definição de escopo, e é a que interessa clinicamente.

## Ética

Os laudos do INbreast são texto clínico. Não reproduza laudos completos em
publicação; use trechos curtos. A existência dos laudos em português do INbreast não é
documentada na literatura — descreva-a como inspeção direta do pacote, nunca citando o
artigo de 2012 como fonte.

## Nota de escopo (24/08/2026)

O CDD-CESM foi removido do escopo de dados do projeto. O único corpus de laudo em texto
livre disponível agora é o **INbreast (117 laudos reais, em português)** — que passa de
"estudo de caso qualitativo" a **corpus principal de auditoria de laudo**. Isso afeta a
meta de anotação (§7.4 do plano), as camadas de avaliação (§8.3) e os riscos R3/R4 (§10).
Veja a "Nota de revisão" no início de `PLANO_TCC_auditoria_mamografia.md` para o resumo
completo das consequências.
