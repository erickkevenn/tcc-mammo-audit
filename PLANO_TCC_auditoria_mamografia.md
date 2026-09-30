# Verificação multimodal de laudos de mamografia

## Plano de execução do TCC — versão revisada

**Autor:** Erick Alves — Edge/UFAL
**Data deste plano:** 19 de agosto de 2026
**Janela de execução:** 19/08/2026 → 18/12/2026 (17 semanas)
**Hardware assumido:** 1 GPU local com ≤ 8 GB de VRAM + Colab/Kaggle como capacidade de surto

**Título proposto (ajustado):**
> *Auditoria automática de laudos de mamografia: detecção incondicional de achados e verificação de descritores BI-RADS condicionada ao texto*

**As duas perguntas do sistema:**
1. **Omissão** — há na imagem um achado suspeito que o laudo não menciona?
2. **Discordância** — o que o laudo descreve (lado, localização, tipo de achado, descritor, categoria BI-RADS, densidade) confere com a imagem?

**Postura:** auditar, não substituir. A saída do sistema é um **alerta rastreável com evidência visual**, não um diagnóstico.

---

## 0. Veredito: o tema faz sentido?

**Sim — e é melhor do que você imagina, por um motivo diferente do que você provavelmente supõe.**

O que verifiquei na literatura de 2023–2026:

| Componente do seu tema | Situação | Consequência para você |
|---|---|---|
| Cruzar achado detectado na imagem com o que o laudo diz | Já estabelecido em **radiografia de tórax** (IBM, MICCAI 2025; CorBenchX 2025; RADAR 2026) | ❌ Não reivindique como novidade conceitual |
| **Detecção de omissão** ancorada na imagem | O grupo mais avançado (IBM/Syeda-Mahmood, MICCAI 2025) diz textualmente: *"Focamos nos dois primeiros erros"* e *"Trabalho futuro no modelo FC abordará achados omitidos dos laudos"* | ✅ **Continua aberto até em tórax** |
| Desempenho atual da tarefa | Melhor VLM comercial no CorBenchX: **50,6% de acurácia de detecção**. RADAR: melhor *composite score* **0,339**. MammoVQA: a maioria dos VLMs médicos ficou *"estatisticamente equivalente a palpite aleatório"* | ✅ A barra é baixa e mensurável. Um híbrido especializado que a supere em domínio fechado é contribuição defensável |
| Modalidade **mamografia** | Nenhum trabalho de auditoria imagem↔laudo publicado | ✅ Novo |
| Léxico **BI-RADS** como esquema de auditoria | Nada publicado | ✅ Novo |
| **Português brasileiro** | Nada além do BRAX (tórax) e do 4º Desafio de IA da SPR (2026, só texto) | ✅ Novo |
| Justificativa da abordagem **híbrida** (CV especializado + NLP simbólico) em vez de VLM ponta a ponta | Empiricamente sustentada pelos três números acima | ✅ Sua escolha metodológica tem defesa citável |

**Formulação honesta para a banca:**

> A auditoria automática de consistência imagem–laudo foi estabelecida em radiografia de tórax (Syeda-Mahmood et al., MICCAI 2025; CorBenchX, 2025; RADAR, 2026), mas com desempenho ainda insuficiente — 50,6% de acurácia de detecção no melhor modelo de visão-linguagem avaliado — e com a detecção de omissões explicitamente relegada a trabalho futuro. Não há, ao nosso conhecimento, trabalho aplicando este enquadramento à mamografia, ao léxico BI-RADS, nem a laudos em português.

E existe motivação **brasileira e empírica**: Serapião, Suzuki & Azevedo-Marques (*Radiologia Brasileira*, 2010) mineraram **22.247 laudos de mamografia** de um hospital universitário brasileiro e encontraram que apenas **11–61%** continham termos BI-RADS corretamente formatados, **21%** dos termos descritivos tinham erro ortográfico, e a palavra "microcalcificação" apareceu com **36 grafias diferentes**. Esse artigo é o parágrafo de abertura da sua introdução.

### 0.1 O que precisa mudar no seu plano original

Cinco correções. Duas delas são, na minha leitura, fatais se não forem feitas.

**🔴 Fatal 1 — "Geração de laudos sintéticos a partir das categorias BI-RADS nas planilhas" torna a tarefa circular.**
Se o laudo é gerado *da mesma anotação* que serve de padrão de referência para o detector, então o auditor está comparando a anotação consigo mesma. Um sistema que apenas leia o CSV acerta 100%. Não há tarefa científica ali, e uma banca atenta vai perceber.
**Correção:** use **laudos reais** como corpus de auditoria. Você já tem dois na sua pasta e há um terceiro grátis (§1). Laudo sintético entra apenas como *ampliação controlada*, e sempre por **injeção de erro em laudo real**, não por geração a partir do rótulo.

**🔴 Fatal 2 — o VinDr-Mammo não anota achados benignos.**
Verifiquei no seu próprio arquivo: `finding_birads` só assume os valores `BI-RADS 3`, `BI-RADS 4` e `BI-RADS 5`. Não há caixa para achado BI-RADS 1–2. Logo, "omissão de achado" no VinDr só pode significar **omissão de achado suspeito (BI-RADS ≥ 3)**.
**Correção:** declare esse escopo no título da seção de metodologia. Não é limitação vergonhosa — é uma definição de escopo, e é a que interessa clinicamente.

**🟡 Correção 3 — CLAHE não é premissa, é hipótese.**
Nenhuma das soluções vencedoras do desafio RSNA 2023 (1.687 equipes) usa CLAHE. Onde existe ablação isolada, o ganho é de **+0,002 a +0,006 AUC** — dentro do ruído. Os ganhos gigantes atribuídos a "pré-processamento" na literatura (ex.: mAP₅₀ 0,438 → 0,590 no VinDr) vêm de **recorte da mama + resolução efetiva**, não de realce de contraste, porque os artigos aplicam as duas coisas juntas e não separam.
**Correção:** CLAHE vira **um braço de ablação**. E isso é oportunidade: não existe na literatura 2022–2026 uma ablação limpa de CLAHE isolado do recorte em mamografia com detector moderno. Quatro execuções suas preenchem essa lacuna.

**🟡 Correção 4 — "separar 20% para teste" não basta.**
Cada exame tem 4 vistas da mesma paciente, e a mesma lesão aparece em CC e MLO. Split aleatório por imagem **garante** vazamento. Além disso, o VinDr já tem split oficial (4.000/1.000 exames, estratificado) — se você não usá-lo, seus números não são comparáveis a nada publicado. No CBIS-DDSM medi um problema adicional: 13 pacientes de `mass_train` aparecem em `calc_test` e 18 de `calc_train` em `mass_test`.
**Correção:** split oficial do VinDr; split próprio agrupado por paciente onde você fundir subconjuntos.

**🟡 Correção 5 — "prever a classificação BI-RADS visual" dentro do detector é o elo mais frágil.**
O teto realista de BI-RADS 5 classes no VinDr é **macro-F1 ≈ 0,59 / AUROC ≈ 0,77** (perda ordinal EMD, *J. Clin. Med.* 2026). Se você pendurar essa cabeça no detector, ela vira a fonte dominante de falsos alertas do auditor.
**Correção:** separe as tarefas — detector prevê **categoria de achado** (massa, calcificação suspeita, distorção, assimetria…), e um classificador **por mama** prevê BI-RADS e densidade, com **perda ordinal** e agregação explícita imagem → mama. O auditor consome as duas saídas com **níveis de confiança diferentes**.

**➕ Ajuste 6 — cuidado com "detecção condicionada ao laudo" no título.**
Condicionar o detector principal no texto do laudo cria **vazamento circular**: o detector "vê" o que o laudo afirma e tende a concordar. O desenho correto (e é o que o pipeline do IBM faz) tem **dois caminhos**:
- **Caminho A, incondicional:** detector → conjunto de achados candidatos na imagem. Serve para responder *omissão*.
- **Caminho B, condicionado ao texto:** para cada sentença de achado do laudo, tente localizá-la na imagem (*grounding*). Frase sem ancoragem confiável = possível achado inexistente/erro. Serve para responder *discordância*.

Mantenha "condicionada ao laudo" no título — mas deixe explícito no método que o condicionamento atua **na verificação**, não na geração de candidatos.

---

## 1. Inventário real dos seus dados (auditado por mim, em `D:\Erick\Dataset`)

Não confie em descrições genéricas dos datasets; abaixo está o que existe **na sua máquina**, contado por mim.

### 1.1 VinDr-Mammo ✅ completo

`vindr-mammo-...-1.0.0/` com `images/` (**5.000** pastas de estudo), `breast-level_annotations.csv`, `finding_annotations.csv`, `metadata.csv`, `LICENSE.txt`, `SHA256SUMS.txt`.

Números que **contei nos seus CSVs**:

| Item | Valor medido |
|---|---|
| Estudos / imagens | 5.000 / 20.000 (FFDM, 4 vistas) |
| Linhas em `finding_annotations.csv` | 20.486 |
| Linhas **com bounding box** | **2.254** |
| Split oficial | training 16.391 / test 4.095 linhas |
| `breast_birads` (20.000 linhas de nível-mama × 2 vistas) | BI-RADS 1: 13.406 · 2: 4.676 · 3: 930 · 4: 762 · 5: 226 |
| `breast_density` | C: 15.292 · D: 2.700 · B: 1.908 · **A: 100** |
| `finding_birads` | apenas 3 (893), 4 (995), 5 (241) — **nunca 1 ou 2** |
| Categorias com caixa (top) | Mass 1.123 · Suspicious Calcification 402 · Focal Asymmetry 232 · Architectural Distortion 95 · Asymmetry 90 · combinações (`Calc+Mass` 82) · Suspicious Lymph Node 57 · Skin Thickening 38 · Global Asymmetry 24 · Nipple Retraction 12 · Skin Retraction 7 |

**Leituras obrigatórias desses números:**
- **Não existe BI-RADS 0** nos seus dados. Não escreva "0–5" na dissertação.
- **Densidade A tem 100 linhas** (= 50 mamas). É inutilizável como classe própria; funda A+B ou reporte como classe degenerada.
- **`finding_categories` é multi-rótulo** e vem como *string* de lista Python (`"['Mass']"`). Parseie com `ast.literal_eval`, não com `split(',')`.
- Só **4,94%** das mamas são BI-RADS 4–5. Todo o seu planejamento de métrica tem de assumir classe positiva rara.
- **Este é o seu dataset de treino do módulo de visão.** É FFDM, é grande, tem caixas com BI-RADS por achado e split oficial revisado por dupla leitura com arbitragem.

### 1.2 CBIS-DDSM ✅ completo (com os CSVs)

`cbis_ddsm/` com **6.672 entradas**: **3.261** pastas de imagem completa (sem sufixo `_N`), 3.410 pares *cropped* + máscara ROI, e `metadata/` com os quatro CSVs oficiais (`mass_case_description_{train,test}_set.csv`, `calc_case_description_{train,test}_set.csv`) + `metadata.csv`.

**Papel no TCC: pré-treino e teste de robustez — NÃO treino principal, NÃO avaliação principal.**
É **filme digitalizado** (42–50 µm), não FFDM. O *domain gap* é medido: treinar em FFDM (INbreast+VinDr) e testar em CBIS-DDSM derruba mAP₅₀ de ~0,96/0,59 para **0,447**; treinar em CBIS-DDSM (filme) e testar em FFDM chinês (CMMD) derruba AUC de 0,867 para **0,568**. Rótulos benignos do DDSM dependem de opinião do radiologista, sem histologia.

Armadilhas que medi/verifiquei e que você vai encontrar:
- Massas usam a coluna `breast_density` (underscore); calcificações usam `breast density` (espaço). Normalize antes de concatenar.
- `assessment` contém valor **0** (incompleto) — decida explicitamente se exclui.
- Existem `subtlety = 0` e `breast density = 0`, fora de escala.
- As colunas de caminho **trocam** *cropped* e máscara de forma inconsistente entre linhas. **Não confie nelas.** Regra robusta: dentro da pasta `..._N`, leia os dois DICOMs e classifique por conteúdo — a **máscara é binária** (2 valores, tipicamente {0,255}) e tem as dimensões da imagem completa; o *cropped* é multivalorado e menor.
- Vazamento cruzado massa↔calcificação nos splits oficiais (13 e 18 pacientes). Se fundir, refaça o split por paciente.

### 1.3 INbreast ✅ completo — **e você tem um ativo que a literatura não documenta**

`INbreast Release 1.0/` com `AllDICOMs/` (**410** `.dcm`, 14 bits, 70 µm, 3328×4084 ou 2560×3328), `AllXML/` + `AllROI/` (contornos OsiriX), `PectoralMuscle/`, `INbreast.csv` / `.xls` (BI-RADS e ACR **por imagem**), e — o ponto importante —

> **`MedicalReports/` com 117 arquivos `.txt` de laudos reais, em PORTUGUÊS, contendo categoria BI-RADS explícita.**

Amostras verbatim que li dos seus arquivos:

```
Doente observada a 2009 02 com antecedentes de tumorectomia direita,
O estudo imagiológico documenta nódulo com distorção do estroma, localizado no QII da mama
direita com 2 cm de diâmetro, sem microcalcificações associadas, de achados suspeitos de malignidade.
Efectuou-se microbiopsia ecoguiada onde foram colhidos 4 fragmentos, para estudo anatomopatológico.
Axila negativa.
Alterações com diagnóstico histológico de malignidade - Bi-Rads - 6.
```

```
Nódulo palpável com cerca de 2cm de diâmetro à direita, TQS, suspeito.
MB- 4 fragmentos.
Axila normal.
Alterações com suspeição elevada de malignidade - Bi-Rads - 4c
```

**Por que isso é relevante e delicado ao mesmo tempo:**
- ✅ São laudos **em português**, com léxico BI-RADS (`Bi-Rads - 4c`, `Bi-Rads - 6`), quadrantes (`QII`, `TQS` = *transição dos quadrantes superiores*), descritores (`nódulo`, `distorção do estroma`, `microcalcificações`), e negação (`sem microcalcificações associadas`, `Axila negativa`).
- ⚠️ **A literatura não documenta esses laudos.** O artigo original de 2012 (*Academic Radiology*), o anúncio na lista `mammoimage` e duas revisões de datasets de 2023 afirmam que o INbreast **não** traz laudos textuais. Nenhuma publicação que localizei os usa.
- 👉 **Consequência prática:** isso é potencialmente um **achado original do seu TCC** — mas você **não pode** citar o artigo de 2012 como fonte da existência dos laudos, porque ele não diz isso. Descreva-os como inspeção direta do pacote `INbreast Release 1.0`, com contagem exata (117 arquivos), *encoding* (Latin-1/CP1252 — os seus estão com acentuação quebrada em UTF-8; leia com `encoding='latin-1'`), estrutura e chave de junção.
- ⚠️ **Cuidado ético.** A licença/consentimento do INbreast não é explícita sobre redistribuição de texto clínico. **Não** reproduza laudos completos no corpo do TCC; use trechos curtos anonimizados, como fiz acima, e discuta isso na seção de ética.
- ⚠️ **Cuidado metodológico.** São **117 laudos para 410 imagens / 110 pacientes** → laudo é por **exame/paciente**, não por imagem. Alguns nomes têm sufixo de data (`069212ec65a94339_2009_01.txt`), ou seja, **exames longitudinais do mesmo paciente**. E os laudos descrevem **também ultrassom, biópsia e axila** — informação que **não está na mamografia**. Se o seu auditor sinalizar "o laudo menciona axila negativa e eu não detectei nada", isso é falso alerta por construção. **Filtre o laudo por seção/modalidade antes de auditar.**

### 1.4 CDD-CESM ⚠️ parcial — **e é o dataset que deve ser o centro da sua avaliação**

O que você tem: `Low energy images of CDD-CESM/` (**1.003** JPEG, as imagens DM ≈ mamografia convencional), `Radiology_hand_drawn_segmentations_v2.csv` (**VIA/VGG**, ~2.971 regiões em 1.233 imagens), e `Medical-reports-for-cases-/Medical reports for cases/` com **327** `.docx` (326 válidos + 1 temporário `~$P54.docx` a filtrar).

Exemplo real que li do seu `P1.docx`:

```
PATIENT NO.1
DIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED:
ACR C: Heterogeneously dense breasts.
Left Breast:
Upper architectural distortion is seen.
No suspicious microcalcifications.
Normal skin thickness and contour of breast.
OPINION:
Left Breast:
Upper architectural distortion (BIRADS 4).
CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:
Left Breast:
Upper heterogeneous non mass enhancement (BIRADS 4).
```

**Por que este é o coração da sua avaliação:** é o único conjunto público, **CC BY 4.0 e sem credenciamento**, que reúne simultaneamente (i) imagem, (ii) contorno de lesão desenhado por radiologista, (iii) descritores BI-RADS estruturados por imagem e (iv) **laudo em texto livre estruturado, com achados e BI-RADS por mama**. Ou seja: você constrói padrão de referência de **omissão** e de **discordância de descritor** cruzando (ii)/(iii) com (iv), **sem depender de injeção sintética** — e usa injeção sintética só para controlar dificuldade e ampliar N.

**O que falta baixar (2 minutos, download direto, sem login):**

```
https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx
```

Essa planilha (239 KB, 10 abas) é o que fecha o ciclo: aba `all` com 2.006 linhas (`Image_name, Patient_ID, Side, Type, Age, Breast density (ACR), BIRADS, Findings, View, Tags, Machine, Pathology`), e abas por tipo de achado com **exatamente os descritores BI-RADS que seu extrator precisa comparar**: `mass_description` (310 linhas, com `Mass shape`, `Mass margin`, `Mass density`), `mass enhancement_description` (334), `asymmetry` (231), `calcifications` (240), `nonmass enhancement_description` (184), `postoperative` (159), `distortion` (50), `postneoajuvant chemotherapy` (8).

Armadilhas verificadas do CDD-CESM:
- BI-RADS aparece com **valores compostos separados por `$`** (`3$2`, `4$2`, `5$2`, até `3$2$2`) = múltiplos achados na mesma imagem. Trate `$` como separador.
- Densidade ACR só é preenchida nas linhas **DM**; nas linhas CESM vem `_`. Faça *forward-fill* por paciente/lado.
- O CSV de segmentação **não é só polígono**: polygon 2.028, ellipse 598, circle 331, polyline 10, point 4. Um conversor que só trate polígonos **perde ~32% das regiões**.
- Erros ortográficos consistentes no corpus (`speculated` por *spiculated*, `Heterogenously`). Seu extrator por regra tem de tolerar.
- Você tem só as imagens **DM (low energy)**, não as subtraídas CESM. Isso é **bom** para o seu escopo: DM é o análogo direto da mamografia convencional. Mas o laudo tem uma seção CESM que você **precisa descartar antes de auditar** — auditar a seção "CONTRAST ENHANCED..." contra uma imagem DM é comparar coisas diferentes. Esse filtro é uma das primeiras coisas a implementar.

### 1.5 O que vale a pena acrescentar

| Dataset | Por quê | Custo | Prioridade |
|---|---|---|---|
| **`Radiology-manual-annotations.xlsx`** (CDD-CESM) | Fecha o padrão de referência de descritores | 239 KB, download direto | 🔴 **Faça hoje** |
| **DMID** (figshare, CC BY 4.0) | **510 imagens + 510 laudos `.txt`** em inglês, 1 laudo por imagem, com `Breast Composition`, `BIRADS`, `Findings`. É o único par imagem↔laudo mamográfico usado como benchmark publicado (**AMRG**, arXiv 2508.09225, split 407/51/52) | ~11,7 GB total, mas `Reports.zip` tem **0,2 MB** — baixe só os laudos + `Metadata.xlsx` se o disco apertar | 🟠 **Alta** — dá comparabilidade com um baseline publicado |
| **Laudos mamográficos em espanhol (Paraguai)** — Zenodo `10.5281/zenodo.14827680` | **4.357 laudos** com BI-RADS e seções segmentadas. Língua românica, léxico BI-RADS quase decalcado do português. Ótimo para prototipar e validar o extrator PT antes de ter volume em português | pequeno, aberto | 🟡 Média |
| **4º Desafio de IA da SPR (JPR 2026)** | Único corpus público de **laudos de mamografia em PT-BR** que localizei (predição de BI-RADS a partir do texto, hospedado no Kaggle) | registro | 🟡 Média — verifique se o dataset segue acessível após o prazo de 28/04/2026 |
| **ReXErr-v1** (PhysioNet, Open Data Commons, **acesso aberto**) | Taxonomia canônica de 12 tipos de erro + exemplos. Use como referência de desenho da sua injeção | 509 MB | 🟡 Média — pode citar só a taxonomia |

---

## 2. Estado da arte: o que usar, o que não usar, e quais números prometer

### 2.1 Números-alvo honestos (se você superar muito isso, procure vazamento antes de celebrar)

| Tarefa | Dataset | SOTA verificado |
|---|---|---|
| Detecção de **massa**, mAP@0.5 | VinDr-Mammo | **0,58–0,65** (Mammo-CLIP 0,58; Mammo-FM 0,58; YOLOv12-L 0,59; MM-DETR multi-vista 0,654) |
| Detecção de **calcificação**, mAP@0.5 | VinDr-Mammo | **0,10–0,35** — colapsa por resolução |
| BI-RADS 5 classes | VinDr-Mammo | macro-F1 **0,593** / AUROC **0,769** (perda ordinal EMD) |
| Calcificação (classificação), AUC | VinDr-Mammo | **0,98** (Mammo-CLIP EN-B5 *fine-tuned*) |
| Massa (classificação), AUC | VinDr-Mammo | **0,88** |
| Densidade, acurácia | VinDr-Mammo | **0,88** |
| Detecção de massa, mAP₅₀ | CBIS-DDSM | **0,566** (YOLOv12-L) |
| Classificação, AUC (protocolo transparente, imagem inteira) | CBIS-DDSM | **0,742** (ResNet-50 @448) |
| Detecção de massa, mAP₅₀ | INbreast | 0,96–0,995 — **saturado, 410 imagens; use só como validação externa** |
| **Detecção de erro em laudo** (imagem+texto) | CorBenchX (tórax) | **50,6%** de acurácia (o4-mini, zero-shot) |
| Revisão de laudo ancorada em imagem | RADAR (TC abdome) | *composite* **0,339** (Gemini-3-Pro) |
| Geração de laudo mamográfico, acurácia BI-RADS | DMID | **0,558** (AMRG, MedGemma-4B + LoRA) |
| Geração de laudo mamográfico, acurácia BI-RADS | VinDr | **0,755** (MammoWise, MedGemma-4B local) |

### 2.2 Decisões arquiteturais e a evidência delas

**Detector: YOLO (v8/v11) é a escolha defensável. Transformer não vence aqui.**
Comparação controlada (Cancers, mesmo pipeline, 640×640): RT-DETR-L teve **mAP@0.5 0,83 vs 0,92 do YOLOv5n** no VinDr e, no dataset clínico, sensibilidade **0,79 vs 0,87** — os autores concluem que o RT-DETR opera "de forma conservadora" e **não o recomendam** para rastreamento. Em outro estudo (*J. Imaging* 11(9):314), o vencedor foi **RTMDet-X (CNN) 0,688 > YOLOv12-L 0,634 > RT-DETR-X 0,626**. O único ganho robusto de transformer vem junto com **multi-vista** (MM-DETR, +0,074 mAP), não do transformer em si.

**O maior ganho disponível é resolução, não arquitetura.**
- Metade da resolução custa **−0,039 AUC** no VinDr (1152×896 → 576×448).
- 100 µm/px é o ótimo; a 150 µm perde-se 0,020–0,060 AUC; a 200 µm, 0,028–0,105.
- Em um dataset medido, **19,1% das caixas anotadas têm menos de 48×48 px** na resolução original. A 640×640 elas viram ~15 px. **É por isso que calcificação a 640×640 dá mAP₅₀ < 0,116.**
- Sistemas de referência: RSNA 1º lugar **2048×1024**; Mirai 1664×2048; Mammo-FM 1520×912; NYU ~2677×1942.

**Recorte da mama é o ganho mais barato de todo o pipeline.**
Ablação limpa em VinDr-Mammo (MamT⁴): sem recorte ROC-AUC **79,6** → com recorte **84,0**; F1 **44,7 → 56,0**. **+4,4 pontos de AUC de graça.**

**Não remova o músculo peitoral.** Nenhuma solução top do RSNA 2023 remove. O análogo em tórax mostra que máscara agressiva **degrada** (AUROC 0,742 → 0,696) enquanto recorte mantém. Remova apenas se fizer módulo de densidade.

**Pré-treino de domínio: use, mas pelo motivo certo.**
Escalar o pré-treino 33× (Mammo-CLIP 25 mil → Mammo-FM 821 mil mamografias) **não melhorou o mAP de detecção de massa** (0,58 → 0,58). O ganho real do Mammo-CLIP é **eficiência de rótulo**: com **10% dos rótulos** ainda atinge mAP 0,43 em massas. Esse é exatamente o seu regime. Note, para honestidade: num benchmark controlado de 4 países (*Sensors* 26(12):3911), **ResNet-50 do ImageNet (AUC 0,867) superou Mammo-CLIP (0,847) e Mammo-FM (0,846)** *in-domain*.

**Multi-vista ajuda, principalmente em mama densa** (+0,074 mAP; **−5,9% de falsos negativos em densidade C**), mas o VinDr **não fornece correspondência de lesão entre CC e MLO**. Um detector de vista única com **agregação explícita imagem → mama** é defensável e é o que Mammo-CLIP e Mammo-FM reportam. Para o seu auditor, a agregação por mama é **obrigatória** de qualquer forma, porque o laudo descreve por mama.

**Extração de laudo: regras são a baseline, LLM local é o sistema.**
- Regras (medspaCy/ConText) têm *recall* razoável mas **precisão ruim na negação**: F1 0,492 (precisão 0,356) vs CAN-BERT 0,777 em 984 laudos.
- LLMs de **pesos abertos** em zero-shot empatam com GPT-4o e destroem regras na rotulagem de achados: **F1 macro 0,926 (Mistral-Large) vs 0,924 (GPT-4o) vs 0,731 (CheXpert de regras)**; em alemão, 0,916 vs 0,748.
- Modelo pequeno funciona: **Qwen1.5-0.5B** com *instruction tuning* atingiu F1 **0,9014** em 14 patologias, superando o CheXpert (0,8864, p=0,01) e empatando com CheXbert (0,9047, p=0,35). **0,5 B de parâmetros.** Isso é excelente notícia para 8 GB.
- Mas cuidado: **LLM genérico não é confiável para atribuir a categoria BI-RADS final**. GPT-4 obteve concordância AC1 de apenas **0,57 (inglês) / 0,50 (italiano) / 0,49 (holandês)** contra **0,91 humano-humano**, e produziu discordâncias com mudança negativa de conduta em **10,6%** dos casos vs 1,5% entre humanos. Extraia a categoria **por regex quando ela está escrita** no laudo; use o modelo apenas para os descritores.
- Fine-tuning importa no regime de **< 2.000 laudos anotados** — exatamente o seu.

**VLM ponta a ponta não resolve o problema. Essa é a sua justificativa.**
MammoVQA (*Nature Communications* 2025, 12 VLMs, 420 mil pares QA): *"a maioria dos LVLMs pré-treinados mostrou desempenho estatisticamente equivalente a palpite aleatório"*. **MedGemma não tem mamografia nem no treino nem na avaliação** (verifiquei os dois relatórios técnicos, 2025 e 2026).

**Que modelos cabem em 8 GB** (com AMP e *batch* pequeno):

| Modelo | Papel | 8 GB? |
|---|---|---|
| YOLOv8s/v11s @ 1024×640, batch 2 + acumulação | detector | ✅ |
| **Mammo-CLIP EN-B2** (EfficientNet-B2) | *backbone* / classificador BI-RADS+densidade | ✅ |
| Mammo-CLIP EN-B5 | idem, melhor | ⚠️ apertado; use no Colab |
| BioBERTpt (< 400 M) | NER/classificação de laudo PT-BR | ✅ trivial |
| Qwen 0.5–4 B, 4-bit, saída JSON restrita | extrator de laudo | ✅ |
| MedGemma-4B + QLoRA | extrator/gerador | ⚠️ inferência 4-bit sim; *fine-tuning* → Colab |
| RadFact (exige Llama-3-70B) | métrica | ❌ substitua por 8 B e **reporte a degradação** |

---

## 3. Reformulação do problema (a parte que a banca vai cobrar)

### 3.1 Unidade de análise

O laudo de mamografia fala **por mama**, não por imagem. Portanto:

```
imagem (CC, MLO) ──agregação──▶ mama (L, R) ──▶ exame (paciente)
                                    ▲
                            unidade de auditoria
```

Todo alerta é emitido no nível **(exame, mama)**, com evidência apontando para uma ou mais **imagens** e caixas.

### 3.2 Representação canônica: `FindingRecord`

O coração do sistema é um único esquema para o qual **os dois lados convergem** — imagem e texto. Sem isso, o auditor vira um emaranhado de `if`.

```json
{
  "exam_id": "P1",
  "laterality": "L",
  "source": "image | report",
  "findings": [
    {
      "category": "mass | suspicious_calcification | architectural_distortion |
                   focal_asymmetry | asymmetry | global_asymmetry |
                   skin_thickening | skin_retraction | nipple_retraction |
                   suspicious_lymph_node | postoperative | no_finding",
      "region": {"quadrant": "UOQ|UIQ|LOQ|LIQ|retroareolar|axillary|unspecified",
                 "vertical": "upper|lower|unspecified",
                 "horizontal": "outer|inner|unspecified"},
      "descriptors": {
        "mass_shape": "round|oval|irregular|null",
        "mass_margin": "circumscribed|obscured|microlobulated|indistinct|spiculated|null",
        "mass_density": "high|equal|low|fat|null",
        "calc_morphology": "amorphous|coarse_heterogeneous|fine_pleomorphic|fine_linear|punctate|null",
        "calc_distribution": "diffuse|regional|grouped|linear|segmental|null",
        "size_mm": 20.0
      },
      "birads": "1|2|3|4|4a|4b|4c|5|6|null",
      "confidence": 0.83,
      "evidence": {"image_id": "...", "bbox": [x1,y1,x2,y2], "sentence_span": [120,187]}
    }
  ],
  "breast_birads": "4",
  "breast_density": "C",
  "meta": {"modality_section": "DM", "language": "pt|en", "extractor": "rules|llm|cnn"}
}
```

Regras de projeto que valem ouro:
- **Todo campo pode ser `null`.** "O laudo não disse" ≠ "o laudo disse que não há". Modele os três estados: **afirmado, negado, ausente**. A maioria dos erros de sistemas assim vem de confundir negado com ausente.
- **Todo achado carrega `evidence`.** Alerta sem rastreabilidade não é auditoria.
- **`modality_section`** existe para descartar as seções CESM (CDD-CESM) e as menções a ultrassom/biópsia/axila (INbreast) antes de comparar.

### 3.3 Taxonomia de alertas (adaptada de CheXprompt / GREEN / CorBenchX ao BI-RADS)

| Código | Alerta | Definição operacional | Gatilho |
|---|---|---|---|
| `A1` | **Omissão de achado suspeito** | Existe achado detectado com confiança ≥ τ e BI-RADS visual ≥ 3 na mama *X*, e o laudo não afirma nenhum achado compatível naquela mama | imagem ⊃ laudo |
| `A2` | **Achado afirmado sem sustentação na imagem** | O laudo afirma achado na mama *X*, e nenhuma região da imagem é ancorada a essa sentença com confiança ≥ τ' | laudo ⊃ imagem |
| `A3` | **Discordância de lateralidade** | Achado afirmado à esquerda, detectado só à direita (ou vice-versa) | lado |
| `A4` | **Discordância de localização** | Categoria concorda, quadrante/região discorda | região |
| `A5` | **Discordância de descritor** | Categoria concorda, mas `mass_margin`, `mass_shape`, `calc_morphology` ou `calc_distribution` divergem do previsto | descritor |
| `A6` | **Inconsistência de categoria BI-RADS** | BI-RADS do laudo difere do previsto em ≥ 2 níveis, ou cruza a fronteira de conduta (≤3 vs ≥4) | conduta |
| `A7` | **Discordância de densidade ACR** | Densidade do laudo difere da prevista em ≥ 2 níveis | densidade |
| `A8` | **Inconsistência interna do laudo** | Achado descrito na seção *Findings* e ausente na *Opinion*; BI-RADS incompatível com o descritor (ex.: "espiculado" com BI-RADS 2) | texto ⊂ texto |

`A8` é barato, não depende do módulo de visão, tem alto valor clínico e **funciona como plano B** se o detector decepcionar (§9).

### 3.4 Severidade (adote a escala de 5 níveis do ReFiSco)

1. sem erro · 2. não acionável · 3. acionável não urgente · 4. urgente · 5. emergente

Mapeamento sugerido: `A6` cruzando ≤3→≥4 e `A1` com BI-RADS visual 5 são **nível 4–5**; `A5` e `A7` são tipicamente **nível 2–3**. Reporte **sensibilidade estratificada por severidade** — recall próximo de 1,0 nos níveis 4–5, tolerância maior nos níveis 2–3. Isso é o que transforma "acurácia de 62%" em um resultado clinicamente interpretável.

---

## 4. Arquitetura do sistema

```
                    ┌───────────────────────────────────────────┐
   DICOM/JPEG ──────▶  M1  PRÉ-PROCESSAMENTO                    │
                    │  metadados → decode → ROI mama → janela   │
                    │  → canonicalizar lado → resize+pad        │
                    │  saídas: PNG16 (treino) + PNG8 (auditoria)│
                    └──────────────┬────────────────────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
┌──────────────────┐   ┌────────────────────┐   ┌───────────────────────┐
│ M2a DETECTOR     │   │ M2b CLASSIFICADOR  │   │ M2c GROUNDING         │
│ YOLOv8/11-s      │   │ por mama            │  │ (condicionado ao      │
│ 1024×640 + tiles │   │ Mammo-CLIP EN-B2    │  │  texto, caminho B)    │
│ → caixas +       │   │ → BI-RADS (ordinal) │  │ → âncora p/ sentença  │
│   categoria      │   │ → densidade ACR     │  │                       │
└────────┬─────────┘   └──────────┬──────────┘  └───────────┬───────────┘
         │  agregação imagem→mama │                          │
         └────────────┬───────────┘                          │
                      ▼                                      │
          FindingRecord(source="image")                       │
                      │                                      │
laudo ──▶ ┌───────────┴─────────────┐                        │
 (.docx/  │ M3 EXTRATOR DE LAUDO    │                        │
  .txt)   │ seccionador → filtro de │                        │
          │ modalidade → regras +   │                        │
          │ LLM local (JSON restrito)│                       │
          └───────────┬─────────────┘                        │
                      ▼                                      │
          FindingRecord(source="report")                      │
                      │                                      │
                      ▼                                      ▼
          ┌──────────────────────────────────────────────────────┐
          │ M4 AUDITOR DETERMINÍSTICO                            │
          │ casamento por (mama, categoria, região) → regras     │
          │ A1–A8 → severidade → alerta com evidência visual     │
          └──────────────────────┬───────────────────────────────┘
                                 ▼
                    relatório de auditoria (JSON + HTML)
```

**Por que o auditor (M4) é determinístico e não um modelo:** rastreabilidade. Cada alerta tem de ser explicável por uma regra citável e um limiar declarado. É isso que diferencia "auditoria" de "mais uma caixa-preta". Além disso, é o único componente que você consegue depurar em dezembro sem retreinar nada.

---

## 5. M1 — Pré-processamento (receita fechada)

### 5.1 Ordem correta e não negociável

`Stored Values → Modality LUT (Rescale) → VOI LUT/janelamento → polaridade (MONOCHROME1)`

Inverter MONOCHROME1 **antes** do janelamento é o bug silencioso mais comum da área.

### 5.2 Pipeline

```
ETAPA 0 — Triagem de metadados (sem ler pixels)
  pydicom.dcmread(path, stop_before_pixels=True)
  registre em manifest.parquet: TransferSyntaxUID, PresentationIntentType,
    PhotometricInterpretation, BitsStored/Allocated, PixelRepresentation,
    RescaleSlope/Intercept, WindowCenter/Width (e VM), VOILUTFunction,
    presença de (0028,3010) VOI LUT Sequence, ImageLaterality, ViewPosition,
    PatientOrientation, PixelSpacing, Manufacturer, PixelPaddingValue
  DESCARTE tudo com PresentationIntentType == "FOR PROCESSING"
  → essa tabela é o que torna o TCC auditável e revela domain shift entre datasets

ETAPA 1 — Decodificação
  roteie por TransferSyntaxUID; use dicomsdl (rápido) com fallback pydicom
  cast para int32 (NÃO int16 — estoura em 16-bit unsigned)

ETAPA 2 — Escala física
  img = img * RescaleSlope + RescaleIntercept
  máscara de padding via PixelPaddingValue/RangeLimit; fallback = moda do histograma

ETAPA 3 — ROI da mama  (o ganho de +4,4 AUC)
  clip p95 → GaussianBlur(5,5) → Otsu → dilate(3,3) → maior contorno → boundingRect
  rejeite se área < 4% do quadro; fallback = quadro inteiro
  (opcional, se sobrar tempo: treinar YOLOX-nano 416 em ~500 imagens anotadas à mão)
  NÃO remova o músculo peitoral

ETAPA 4 — Intensidade (gere DOIS artefatos)
  (a) treino, 16-bit: Winsor p1–p99 DENTRO da máscara da mama → PNG16
  (b) auditoria, 8-bit: janelamento com WC/WW do header (LINEAR/LINEAR_EXACT/SIGMOID)
  if MONOCHROME1: x = MAX - x        ← POR ÚLTIMO
  x[mask_pad] = 0

ETAPA 5 — Geometria
  canonicalize lateralidade (ImageLaterality; fallback = parse de ViewPosition)
  resize preservando aspecto (ratio = min(H/h, W/w)) + padding centralizado com zeros
  alvo massas: 1024×640  (8 GB) | 1536×768 se couber
  alvo calcificações: tiles 512×512 com stride 448 em ~100 µm/px

ETAPA 6 — Normalização
  média/desvio do SEU split de treino se usar pesos ImageNet; z-score por imagem se treinar de zero

ETAPA 7 — Augmentation (albumentations)
  HorizontalFlip(p=0.5)
  ShiftScaleRotate(shift=0.05, scale=0.15, rotate=15, p=0.5)
  RandomBrightnessContrast(0.15, 0.15, p=0.5)
  ElasticTransform(alpha=1, sigma=20, p=0.2)
  SEM MixUp/CutMix. SEM blur. SEM flip vertical. SEM CLAHE fixo (só como braço de ablação).
```

### 5.3 A ablação de pré-processamento (sua contribuição de baixo custo)

Não existe na literatura ablação limpa de CLAHE **isolado do recorte** em mamografia com detector moderno. Cinco execuções resolvem:

| Braço | Configuração | O que testa |
|---|---|---|
| `B0` | Modality LUT → WC/WW → **crop** → resize+pad | baseline |
| `B1` | `B0` com Winsor p1–p99 em vez de WC/WW | normalização robusta |
| `B2` | `B0` + CLAHE(clip=2.0, tile=8×8) | CLAHE "padrão" |
| `B3` | `B0` + CLAHE(clip=12, tile=8×8) | CLAHE "forte" (parâmetro do *Cancers* 2024) |
| `B4` | `B0` **sem crop** | isola o efeito do recorte |

Reporte mAP@0.5 **separado para massa e calcificação**, com IC 95% por bootstrap em nível de exame. Predição: `B4` será claramente o pior e `B2`/`B3` ficarão dentro do IC de `B0`. Se der isso, você tem um resultado negativo publicável e uma seção de discussão inteira.

### 5.4 Custo de disco e tempo (planeje agora)

- VinDr-Mammo em DICOM ocupa ~54 GB; convertido para PNG16 recortado a 1536 px de lado maior, espere **8–15 GB**. CBIS-DDSM em DICOM são ~163 GB — **converta só o que for usar** (imagens completas de casos que você realmente treinar).
- Converta **uma vez** para cache em disco e treine sobre o cache. Grave um `manifest.parquet` com hash do original, parâmetros aplicados e versões de bibliotecas. Isso é o que faz o trabalho ser reprodutível.
- 20.000 imagens com `dicomsdl` + `joblib(n_jobs=n_cores)`: conte com **1,5–4 h** na primeira passada, dependendo do disco. Rode de madrugada, não no dia da apresentação.

---

## 6. M2 — Módulo de visão

### 6.1 Dois caminhos por escala (isso resolve o problema de calcificação)

| | Massas, distorções, assimetrias | Calcificações suspeitas |
|---|---|---|
| Entrada | imagem inteira recortada, 1024×640 | tiles 512×512, stride 448, ~100 µm/px |
| Modelo | YOLOv8s ou YOLOv11s | YOLOv8s treinado só nessa classe |
| Critério de acerto | IoU ≥ 0,5 | **centro dentro da caixa de referência** (IoU é instável em objeto minúsculo) |
| Agregação | NMS por imagem → união por mama | *merge* de tiles por NMS global → cluster por mama |

Justificativa dura: 19,1% das caixas anotadas têm < 48×48 px na resolução original; a 640×640 viram ~15 px. Onde alguém treinou com janelas de 256×256 em alta resolução, **calcificação (F1 0,843) saiu melhor que massa (F1 0,739)** — o inverso do padrão dos detectores em 640×640. **A resolução é o gargalo, não a arquitetura.**

### 6.2 Hiperparâmetros de partida para 8 GB

```yaml
# detector de massas
model: yolov8s            # ou yolo11s
imgsz: [1024, 640]        # h, w após crop e pad
batch: 2
accumulate: 8             # batch efetivo 16
amp: true
optimizer: AdamW
lr0: 0.001
warmup_epochs: 3
epochs: 80                # early stopping com paciência 15 no mAP@0.5 de val
cos_lr: true
cache: false              # 8 GB de VRAM e disco cheio: não cacheie em RAM
workers: 4
patience: 15
# classes: mass, architectural_distortion, focal_asymmetry, asymmetry,
#          global_asymmetry, skin_thickening, skin_retraction,
#          nipple_retraction, suspicious_lymph_node
```

Se estourar VRAM: `imgsz: [896, 576]`, `batch: 1`, `accumulate: 16`. Se sobrar: suba `imgsz` antes de trocar de modelo — o retorno é maior.

**Classes raras:** `skin_retraction` (7 caixas), `nipple_retraction` (12), `global_asymmetry` (24). Com essa contagem, **não** reporte mAP por classe como se fosse mensurável. Duas opções honestas: (a) agrupe em uma superclasse `other_suspicious`; (b) mantenha separadas mas reporte apenas a contagem e diga que o N é insuficiente. Escolha (a) para o sistema e (b) para a tabela do anexo.

### 6.3 Classificador por mama (M2b)

- **Entrada:** as duas vistas da mama (CC + MLO) → *backbone* compartilhado → *pooling* → cabeça. Se a memória apertar, treine por imagem e agregue por `max` da confiança entre CC e MLO.
- ***Backbone*:** Mammo-CLIP **EN-B2** (pesos públicos, EfficientNet-B2). O argumento de defesa está em eficiência de rótulo (mAP 0,43 com 10% dos rótulos), não em SOTA absoluto. Baseline obrigatório: **EfficientNet-B2/ResNet-50 do ImageNet** — no benchmark de 4 países o ImageNet ganhou *in-domain*, então esse baseline pode te surpreender e é honesto reportá-lo.
- **Perdas:** BI-RADS com **perda ordinal (EMD)** — ganho verificado de +0,66 pp de AUROC no VinDr e +1,72 pp no INbreast contra *cross-entropy*, além de reduzir erros severos. Densidade: funda **A+B** (só 50 mamas em A).
- **Métrica:** AUROC por classe, macro-F1 e **kappa ponderado quadrático** (BI-RADS é ordinal).

### 6.4 M2c — *Grounding* condicionado ao texto (caminho B)

Escopo enxuto e realista para um TCC:

1. **Nível 1 (obrigatório, barato):** casamento por região. A sentença do laudo diz "distorção arquitetural no quadrante superior externo esquerdo" → converta para `(L, architectural_distortion, UOQ)` → verifique se alguma caixa detectada cai naquele quadrante da mama esquerda. A conversão caixa → quadrante sai da geometria da máscara da mama (centroide relativo, com o mamilo como referência de "outer").
2. **Nível 2 (se sobrar tempo):** *grounding* por similaridade — projete a sentença no espaço de texto do Mammo-CLIP e compute similaridade com as *features* das regiões candidatas; sentença sem região acima do limiar → candidata a `A2`.
3. **Nível 3 (trabalho futuro, escreva assim):** *grounding* denso estilo *Generate to Ground* (mIoU 0,54 vs 0,27 do BioViL em tórax) e Mammo-FActOR (única coisa próxima de *phrase grounding* em mamografia que localizei — dentro do Mammo-CLIP; **leitura obrigatória**).

Não prometa Nível 3 na qualificação.

---

## 7. M3 — Extrator de laudo

### 7.1 Seccionador (faça primeiro, é o que evita 80% dos falsos alertas)

O CDD-CESM tem estrutura fixa e explorável:
`PATIENT NO.n` → `DIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED:` → blocos `Right Breast:` / `Left Breast:` → `OPINION:` → `CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:` → blocos por mama.

Regras:
- **Descarte tudo a partir de `CONTRAST ENHANCED`** — você só tem imagens DM.
- No INbreast: descarte sentenças sobre **ultrassom, microbiopsia, axila, marcação pré-operatória, histologia**. Um laudo que diz "Axila negativa" não é auditável contra uma mamografia.
- Guarde os *spans* de caractere de tudo o que você extrair (`sentence_span`), para o alerta poder citar o trecho exato.

### 7.2 Léxico BI-RADS bilíngue (PT-BR ↔ EN) — entregável autônomo do TCC

Um YAML versionado, com sinônimos, variantes ortográficas e mapeamento para RadLex quando existir. Isso é **contribuição citável**: não localizei nenhum trabalho publicado de extração de **descritores** BI-RADS granulares em português.

```yaml
mass_margin:
  spiculated:
    pt: [espiculado, espiculada, espículas, espiculadas, com espiculação]
    en: [spiculated, speculated]     # 'speculated' é erro real e frequente no CDD-CESM
  circumscribed:
    pt: [circunscrito, circunscrita, bem definido, bem delimitado, contornos regulares]
    en: [circumscribed, well defined, well-defined]
  indistinct:
    pt: [mal definido, indistinto, contornos imprecisos, mal delimitado]
    en: [indistinct, ill defined, ill-defined]
calc_morphology:
  fine_pleomorphic:
    pt: [pleomórficas, pleomorficas, heterogêneas finas]
    en: [pleomorphic, fine pleomorphic]
region:
  UOQ:
    pt: [quadrante superior externo, QSE, QSL, união dos quadrantes superiores, TQS]
    en: [upper outer quadrant, UOQ, upper outer]
```

Cuidados que os dados **exigem**: `microcalcificação` com 36 grafias documentadas em laudos brasileiros; `speculated`/`Heterogenously` no CDD-CESM; abreviações portuguesas de quadrante (`QII`, `QSE`, `TQS`) que não existem em nenhum léxico pronto.

### 7.3 Três extratores, comparados (é isso que faz uma seção de resultados)

| # | Extrator | Custo | Papel |
|---|---|---|---|
| `E0` | **Regex + léxico + ConText/negação** | horas | **Baseline obrigatório.** Espere precisão ruim na negação (F1 ~0,49 na literatura) |
| `E1` | **LLM local 4 B em 4-bit, JSON restrito** (gramática GBNF / Outlines / `guided_json`) | dias | Sistema principal. Zero-shot já compete com GPT-4o em rotulagem de achado (0,926 vs 0,924) |
| `E2` | **BioBERTpt afinado** para NER de descritores (PT) ou Qwen-0.5B com *instruction tuning* | dias + anotação | Melhor no regime < 2.000 laudos. Precedente: Qwen1.5-0.5B com F1 0,9014 batendo CheXpert |

Regras de honestidade:
- **A categoria BI-RADS escrita no laudo extraia por regex**, não por LLM. GPT-4 tem AC1 de apenas 0,57 nessa tarefa contra 0,91 humano-humano. É um número que **você deve citar** para justificar a escolha.
- Se usar LLM para **injetar** erro e LLM para **detectar**, use **modelos e prompts diferentes** e diga isso no texto. Caso contrário há vazamento de viés de geração.

### 7.4 Anotação: quanto você precisa mesmo

Alvo mínimo defensável: **150 laudos anotados** (100 CDD-CESM em inglês + 50 INbreast em português) com o esquema `FindingRecord` completo. Use **dois anotadores** (você + um colega ou estudante de medicina) em **50 laudos sobrepostos**, e reporte **kappa de Cohen**. Duplo anotador em amostra parcial é barato e é o que separa um TCC sério de um TCC "eu anotei tudo sozinho".

---

## 8. M4 — Auditor e avaliação

### 8.1 Casamento e tolerâncias (declare os limiares ANTES de rodar o teste)

```python
MATCH = {
  "laterality":  "exato",
  "category":    "exato, com fusão de sinônimos (asymmetry ⊃ focal/global)",
  "region":      "adjacência permitida: UOQ~upper, UOQ~outer; 'unspecified' casa com tudo",
  "size_mm":     "tolerância ±30% ou ±5 mm, o que for maior",
  "birads":      "discordância se |Δ| ≥ 2 níveis OU cruza a fronteira ≤3 / ≥4",
  "density":     "discordância se |Δ| ≥ 2 níveis",
}
TAU_DETECT   = 0.35   # calibrado no split de validação, NUNCA no teste
TAU_GROUND   = 0.25
```

Calibre `TAU_DETECT` para o ponto de operação **sensibilidade @ 0,5 FP/imagem** no conjunto de validação, e **congele-o** antes de tocar no teste. Escreva no caderno de laboratório a data em que você congelou. Bancas gostam disso.

### 8.2 Métricas

**Módulo de visão (primária):** sensibilidade em nível de lesão @ **0,25 / 0,5 / 1,0 FP por imagem**, com curva **FROC** completa e IC 95% por bootstrap **reamostrando exames** (nunca imagens nem lesões — CC e MLO da mesma mama são correlacionadas). Secundárias: mAP@0.5 (comparabilidade com a literatura) e F1 no ponto de operação escolhido.

Justificativa citável (*Metrics Reloaded*, Maier-Hein et al., *Nature Methods* 2024): para detecção, combine uma métrica de contagem (F-beta) com uma métrica multi-limiar (AP para padronização **ou** FROC para interpretabilidade), e escolha o critério de localização conforme a granularidade clínica. O próprio framework critica reportar uma única métrica num corte arbitrário.

**Auditor (primária):** precisão e *recall* **por tipo de alerta** (`A1`–`A8`), mais **sensibilidade estratificada por severidade** (1–5).

**Baselines obrigatórios** — sem eles, nenhum número tem interpretação:
- `Z0` **"sempre sem erro"** → mostra o *prior*. Se 90% dos laudos estão corretos, um sistema burro acerta 90%.
- `Z1` **lexical** → busca palavra-chave do achado no laudo, sem imagem.
- `Z2` **VLM zero-shot** (Qwen2.5-VL-7B ou MedGemma-4B em 4-bit, imagem + laudo, pergunta direta) → sua comparação com o estado da arte. A barra publicada é **50,6%** (CorBenchX).

**Composite Score** (copiado do RADAR): exige acerto simultâneo em (i) existe erro, (ii) qual o tipo, (iii) qual a severidade. É mais duro e mais informativo que acurácia binária.

### 8.3 Como construir o conjunto de teste de auditoria

**Camada 1 — erros naturais (o mais valioso).** No CDD-CESM, cruze a planilha `Radiology-manual-annotations.xlsx` (descritores + BI-RADS por imagem) com os 326 `.docx`. **Toda divergência real é um caso de auditoria não sintético.** Depois, revise manualmente uma amostra para separar divergência genuína de artefato da sua própria extração — esse passo não é opcional, é o que dá validade ao número.

**Camada 2 — injeção sintética controlada** em laudos reais, com taxonomia adaptada ao BI-RADS:

| Perturbação | Como | Análogo na literatura |
|---|---|---|
| Omissão | remover a sentença de achado que a anotação confirma existir | `Omission` (CorBenchX, 6.267 casos) |
| Troca de lateralidade | L↔R | `Side Confusion` (a **maior** categoria do CorBenchX, 7.615) |
| Mudança de severidade | BI-RADS 4c→3, 5→4a; "suspicious"→"benign" | `Change Severity` (ReXErr) |
| Mudança de localização | quadrante, retroareolar | `Change Location` (ReXErr) |
| Achado fabricado | inserir massa espiculada em mama descrita como normal | `False Prediction` (ReXErr) |
| Mudança de descritor | `spiculated`→`circumscribed`, `irregular`→`round`, `pleomorphic`→`punctate` | — (**seu diferencial**) |
| Mudança de densidade | ACR C→A | — |

**Regra do ReXErr, obedeça:** perturbação **condicionada ao contexto**. Não injete "troca de margem de massa" num laudo sem massa. Um erro impossível é ruído, não avaliação.

**Camada 3 — validação de plausibilidade.** Mínimo aceito pela literatura: revisão clínica de **≥ 100 pares** injetados, reportando taxa de plausibilidade (ReXErr: **83/100**). **Vá além:** use **2 avaliadores** e reporte **kappa de Cohen** — o revisor único é a limitação explícita deles e é barato de superar.

### 8.4 Estatística com N pequeno (leia isto antes de escrever conclusões)

- **Bootstrap** reamostrando **exames**, 2.000+ reamostras, método percentil ou **BCa** (melhor para métricas assimétricas como AP e F1 em classe rara).
- **Realismo brutal:** para um IC de 1% de largura em métrica contínua por caso são necessários ~100–200 casos; para **acurácia** (métrica discreta), ~**10.000**. Com 50–300 exames você **não** terá IC estreito. Planeje para isso: (i) reporte a largura do IC como resultado, não como incômodo; (ii) escreva "não foi possível descartar diferença", nunca "os modelos são equivalentes"; (iii) faça **comparações pareadas** no mesmo conjunto de teste — ganham muito poder.
- **DeLong** para comparar AUCs correlacionadas; **McNemar** (versão exata se b+c < 25) para acurácia pareada num ponto de operação; **kappa ponderado quadrático** para BI-RADS; **Fleiss** para ≥3 avaliadores. Interprete kappa pelos limiares de **Landis & Koch (1977)** citando a fonte e reportando também concordância bruta e prevalências marginais (kappa colapsa com prevalência desbalanceada).
- **Precedente de tamanho:** o RADAR (2026) publicou um benchmark com **50 casos de teste** (20 val / 30 teste). Você não precisa de milhares para ter um benchmark defensável — precisa de rigor no desenho.
- **Múltiplas comparações:** declare **uma** métrica primária e **um** teste primário no protocolo. O resto é exploratório, ou aplique Holm-Bonferroni / Benjamini-Hochberg e diga qual usou.

### 8.5 Checklist de relato

Use **CLAIM 2024** (*Checklist for AI in Medical Imaging*, 44 itens, Tejani et al., *Radiology: AI* 6(4), DOI `10.1148/ryai.240300`) como checklist primário, preenchido e anexado. Pontos que ele cobra e que mudam seu texto:
- Diga **"padrão de referência"**, não "ground truth".
- Não use a palavra ambígua "validação": diga **teste interno** vs **teste externo**. Treinar em VinDr e testar em CDD-CESM/DMID **é teste externo** — declare isso, é um ponto forte do seu desenho.
- **Item 13** exige documentar protocolo de aquisição: FFDM vs filme digitalizado, fabricante, resolução, *bit depth*. Sua tabela de triagem de metadados (§5.2, Etapa 0) já responde isso.
- Se o módulo de laudo for LLM, complemente com **TRIPOD-LLM**. **Não** anexe três checklists.

---

## 9. Cronograma semanal (19/08 → 18/12/2026)

**Princípio de execução: fatia vertical primeiro.** Seu plano original era em cascata — pré-processar tudo em setembro, treinar em outubro, integrar em novembro. O risco disso é chegar em 20 de novembro sem nunca ter visto o sistema completo funcionando. A ordem abaixo entrega um **pipeline fim-a-fim feio mas completo até a semana 3**, e depois melhora cada peça. Se algo der errado, você sempre tem algo para defender.

### Fase A — Fatia vertical (S1–S3 · 19/08 a 08/09)

| Semana | Entregar | Critério de aceite |
|---|---|---|
| **S1** 19–25/08 | Repositório versionado; ambiente CUDA reprodutível (`requirements.txt` com versões travadas); baixar `Radiology-manual-annotations.xlsx` e os laudos do DMID; parser dos 4 CSVs do CBIS-DDSM e dos 2 do VinDr; **script de split por paciente** | `python -m src.data.build_manifest` roda e gera `manifest.parquet` sem exceção; zero paciente em dois splits (teste automatizado) |
| **S2** 26/08–01/09 | Conversor DICOM→PNG16/PNG8 com ROI da mama (Otsu + maior componente); rodar em **300 imagens** do VinDr como amostra; inspeção visual de 30 recortes | 0 recorte com área < 4% do quadro; nenhuma imagem invertida (checagem de MONOCHROME1); PNG8 legível |
| **S3** 02–08/09 | **Fatia vertical:** detector YOLOv8s treinado 10 épocas em ~1.500 imagens do VinDr + extrator `E0` de regras em 20 laudos do CDD-CESM + auditor com as regras `A1` e `A3` + relatório HTML com caixa desenhada | Um comando produz um alerta de omissão real, com imagem e trecho do laudo, para pelo menos 1 caso |

> **Marco M1 (08/09): o cano está furado mas tem água saindo nas duas pontas.** A partir daqui todo o resto é melhoria incremental, não construção.

### Fase B — Dados e módulo de visão (S4–S8 · 09/09 a 13/10)

| Semana | Entregar |
|---|---|
| **S4** 09–15/09 | Conversão completa do VinDr (20.000 imagens) com cache em disco + `manifest.parquet`; conversão dos 1.003 JPEG DM do CDD-CESM; conversor VIA → bbox tratando **as 5 formas** (polygon/ellipse/circle/polyline/point) |
| **S5** 16–22/09 | Treino sério do detector de massas/distorções/assimetrias, split oficial do VinDr, 1024×640, *early stopping*; primeira curva **FROC** |
| **S6** 23–29/09 | Caminho de **calcificações** por tiles 512×512 em ~100 µm/px; critério de acerto por centro; FROC separada |
| **S7** 30/09–06/10 | **Ablação de pré-processamento** `B0`–`B4` (5 execuções); tabela com IC 95% bootstrap por exame |
| **S8** 07–13/10 | Classificador por mama (BI-RADS ordinal EMD + densidade A+B fundidas), Mammo-CLIP EN-B2 vs EfficientNet-B2 ImageNet; agregação imagem→mama; kappa ponderado quadrático |

> **Marco M2 (13/10): módulo de visão congelado.** Anote no caderno: pesos, *commit*, métricas, limiares. Não mexa mais, exceto se M4 provar que é o gargalo.

### Fase C — Laudo e auditoria (S9–S13 · 14/10 a 17/11)

| Semana | Entregar |
|---|---|
| **S9** 14–20/10 | Seccionador (CDD-CESM: descartar seção CESM; INbreast: descartar US/biópsia/axila); léxico BI-RADS bilíngue v1 em YAML; extrator `E0` completo com negação |
| **S10** 21–27/10 | **Anotação:** 150 laudos com o esquema `FindingRecord` (100 EN + 50 PT), 50 em duplicata com um segundo anotador → kappa de Cohen |
| **S11** 28/10–03/11 | Extrator `E1` (LLM 4 B local, 4-bit, JSON restrito por gramática) + comparação `E0` vs `E1` vs (`E2` se houver tempo) no conjunto anotado |
| **S12** 04–10/11 | Auditor completo `A1`–`A8` com severidade; calibração dos limiares **no split de validação**; relatório HTML com evidência visual |
| **S13** 11–17/11 | **Benchmark de auditoria:** camada 1 (erros naturais CDD-CESM) + camada 2 (injeção sintética, 7 perturbações, condicionada ao contexto) + camada 3 (validação de plausibilidade com 2 avaliadores, ≥100 pares, kappa) |

> **Marco M3 (17/11): benchmark de auditoria pronto e congelado.** O conjunto de teste não muda mais. Nem um caso.

### Fase D — Avaliação e escrita (S14–S17 · 18/11 a 18/12)

| Semana | Entregar |
|---|---|
| **S14** 18–24/11 | Avaliação final: sistema completo + baselines `Z0`/`Z1`/`Z2` no teste congelado; bootstrap, McNemar pareado, Composite Score; teste **externo** (treino VinDr → teste CDD-CESM / DMID) |
| **S15** 25/11–01/12 | Análise de erro: 30 falsos positivos e 30 falsos negativos categorizados manualmente por causa (falha do detector / falha da extração / ambiguidade genuína do laudo / erro do padrão de referência). **Esta tabela costuma ser a melhor seção do TCC** |
| **S16** 02–08/12 | Redação: Metodologia + Resultados + Discussão; preencher o checklist **CLAIM 2024** (44 itens) |
| **S17** 09–15/12 | Redação: Introdução + Revisão da literatura + Conclusão; revisão do orientador; figuras finais; anexos |
| 16–18/12 | Folga de segurança. **Não planeje trabalho aqui.** Algo vai atrasar. |

> **Marco M4 (15/12): TCC escrito.**

### Regra de ouro do cronograma

Escreva **desde a semana 4**, não desde a semana 16. Toda semana, 45 minutos: cole os números novos numa tabela do documento e escreva o parágrafo que os explica. A Metodologia inteira pode ser escrita em setembro/outubro, porque ela descreve o que você está fazendo naquele momento — e escrever obriga a explicitar decisões que de outro modo ficam implícitas no código.

---

## 10. Riscos e planos B (leia antes de precisar)

| # | Risco | Probabilidade | Sinal precoce | Plano B |
|---|---|---|---|---|
| R1 | **8 GB não bastam no `imgsz` escolhido** | Alta | `CUDA out of memory` na S3 | `imgsz` 896×576, `batch 1` + `accumulate 16`, AMP, `cache=false`; treino longo no Colab/Kaggle (T4 16 GB) com *checkpoint* a cada época e retomada |
| R2 | **Detector de calcificação fica ruim (mAP < 0,15)** | **Alta** — é o padrão da literatura | FROC da S6 | Restrinja o escopo formal a **massa, distorção e assimetria**; declare calcificação como trabalho futuro citando o mAP 0,10–0,35 da literatura. **Isto não invalida o TCC** |
| R3 | **Extração de laudo em português tem precisão baixa** | Média | kappa/F1 da S11 | Faça o sistema principal em **inglês** (CDD-CESM + DMID) e o português como **estudo de caso qualitativo** nos 117 laudos do INbreast, com análise de erro. Continua sendo o primeiro trabalho em PT-BR |
| R4 | **Erros naturais do CDD-CESM são raros demais** (o problema estatístico central) | **Alta** | contagem na S13 | Precedente publicado: em 766 pares de radiologia↔patologia havia **apenas 13 discordâncias reais**, e a AUC de 0,999 reportada era instável. Reporte a raridade **como resultado**, e faça a avaliação quantitativa principal sobre a **camada sintética**, com a natural como validação qualitativa |
| R5 | **Não achar um radiologista para validar plausibilidade** | Média | S12 | 2 estudantes de medicina + adjudicação por um terceiro; declare a limitação explicitamente. É o que o ReXErr fez com **1 único revisor** — você já supera o precedente |
| R6 | **Disco insuficiente** (VinDr 54 GB + CBIS-DDSM 163 GB + cache PNG) | Média | S4 | Não converta o CBIS-DDSM inteiro. Cache PNG16 recortado a 1536 px cabe em 8–15 GB para o VinDr todo. Mantenha os DICOM em HDD e o cache PNG em SSD |
| R7 | **Atraso acumulado empurra a escrita para dezembro** | **Alta** — é o modo de falha mais comum de TCC | S12 sem Metodologia escrita | A regra dos 45 min/semana da §9. E o corte de escopo pré-aprovado: R2 (sem calcificação) + R3 (inglês como principal) reduzem ~3 semanas sem tocar na contribuição central |
| R8 | Aparecer trabalho publicado com a mesma ideia em mamografia | Baixa | alerta de busca | Configure alertas para `mammography report error detection`, `BI-RADS report consistency`, `mammography report audit`. Se aparecer, você tem um baseline para comparar — o que **melhora** o TCC |

**Corte de escopo pré-aprovado (decida agora, não em dezembro).** Se em 17/11 (M3) você não tiver o benchmark pronto, entregue o TCC com: detector de massas apenas + extrator `E0`+`E1` + auditor `A1`,`A3`,`A6`,`A8` + avaliação só na camada sintética. Isso ainda é uma contribuição inédita e defensável. **Escreva esse parágrafo hoje e mostre ao orientador** — decisão de escopo tomada com antecedência é planejamento; tomada na última semana é desespero.

---

## 11. Repositório inicial

Está no ZIP que acompanha este plano — **61 testes passando**, sem precisar de GPU nem dos
dados. Os parsers foram executados contra os seus arquivos reais e reproduzem exatamente os
números da §1. Estrutura:

```
tcc-mammo-audit/
├── README.md · requirements.txt · pytest.ini · .gitignore
├── PLANO_TCC_auditoria_mamografia.md      # este documento
├── configs/
│   ├── paths.yaml                  # ÚNICO arquivo com caminho absoluto
│   ├── preprocess.yaml             # braços B0–B4 da ablação
│   ├── detector_mass.yaml          # YOLO 1024×640, batch 2 + acumulação 8
│   ├── detector_calc.yaml          # tiles 512, stride 448
│   └── audit_rules.yaml            # limiares e tolerâncias A1–A8
├── lexicon/birads_lexicon.yaml     # descritores PT-BR ↔ EN (entregável autônomo)
├── scripts/demo_end_to_end.py      # fatia vertical SEM GPU: laudo → auditoria → HTML
├── src/
│   ├── config.py
│   ├── data/       build_manifest · vindr · cbis · cdd_cesm · inbreast · splits
│   ├── preprocess/ dicom_io · breast_roi · to_yolo
│   ├── vision/     train_detector · train_breast_classifier · tiles · aggregate
│   ├── report/     schema · sectionizer · extract_rules · extract_llm
│   ├── audit/      rules · matcher · report_html
│   ├── eval/       froc · bootstrap · stats · error_injection
│   └── baselines/  z0_always_clean · z1_lexical · z2_vlm_zeroshot
├── notebooks/01_inspect_datasets.ipynb
└── tests/          9 arquivos, 61 testes
    ├── test_splits_no_leakage.py    # vazamento paciente entre splits
    ├── test_bbox_mapping.py         # crop→resize→pad: erre aqui e o mAP mente
    ├── test_via_shapes.py           # as 5 formas VIA do CDD-CESM
    ├── test_schema.py               # NEGATED != ABSENT
    ├── test_audit_rules.py          # uma linha por regra A1–A8
    ├── test_error_injection.py      # condicionamento ao contexto
    ├── test_froc_and_stats.py       # FROC, bootstrap, DeLong, McNemar, kappas
    ├── test_report_extraction.py    # nos formatos REAIS de CDD-CESM e INbreast
    └── test_matcher_geometry.py     # bbox → quadrante
```

Primeiros comandos:

```bash
pip install -r requirements.txt
export TCC_DATA_ROOT="D:/Erick/Dataset"
pytest                                              # 61 testes, ~1 s
python -m src.data.vindr                            # confere os números da §1.1
python scripts/demo_end_to_end.py --exam P1 --side L   # auditoria real, sem GPU
```

O `demo_end_to_end.py` já produz, com um laudo real do CDD-CESM e uma saída de detector
simulada, o alerta `[A1] sev=5 — Achado 'mass' detectado na mama L (conf=0.91, BI-RADS
visual=5) não mencionado no laudo`, mais o HTML de evidência e um mini-benchmark sintético.
Isso é o marco M1 da semana 3 antecipado: o contrato entre os módulos está validado antes
de existir modelo treinado.

**Ambiente (versões verificadas no PyPI em 19/08/2026):**

```
torch==2.13.0            pydicom==3.0.2            dicomsdl==0.109.4
ultralytics             opencv-python-headless==5.0.0.93
albumentations==2.0.8   timm==1.0.28              scikit-image==0.26.0
pandas                  pyarrow                   python-docx
pydantic                openpyxl                  pylibjpeg-openjpeg==2.5.0
```

Atenção: `numpy>=1.24` removeu `np.int`/`np.float` — muito código de mamografia publicado (inclusive o script oficial de conversão do VinDr) quebra por isso. E o `albumentations` só aplica CLAHE em `uint8`/`float32`; para 16-bit use `cv2.createCLAHE`.

**Código de referência que vale reaproveitar** (leia, não copie cego):
- `dangnh0611/kaggle_rsna_breast_cancer` — 1º lugar do RSNA 2023. O mais valioso: `src/utils/dicom.py` (6 *backends* de decodificação), `src/utils/windowing.py` (LINEAR/LINEAR_EXACT/SIGMOID em numpy e torch), `src/roi_det/roi_extract.py` (YOLOX + *fallback* Otsu).
- `vinbigdata-medical/vindr-mammo` — `visualize.py` é a conversão oficial do dataset. **Dois bugs conhecidos:** `a[a>MM] = 255` executa antes de calcular a máscara (contamina a máscara quando `MM < 255`), e usa `np.int` (removido). Corrija ao reusar.
- `batmanlab/Mammo-CLIP` — pesos EN-B2/EN-B5, *linear probe*, *fine-tuning* e detecção com RetinaNet já implementados; treino em GPU única.
- `reginabarzilaygroup/Mirai` — `onconet/utils/dicom.py` é a melhor referência de tratamento por fabricante (GE com VOI LUT Sequence, Hologic C-View com WC=540/WW=580 fixos, *fallback* min-max).
- `Parvaneh-Aliniya/pectoral_muscle_groundtruth_segmentation` — máscaras de peitoral para INbreast/CBIS-DDSM/MIAS, se você fizer o módulo de densidade.

---

## 12. Tabelas que o TCC precisa ter (esqueleto para preencher)

| # | Tabela | Quando |
|---|---|---|
| T1 | Caracterização dos datasets: modalidade (FFDM vs filme), fabricante, resolução, *bit depth*, nº de imagens/pacientes, tipo de anotação, licença | S4 — atende o **item 13 do CLAIM** |
| T2 | Distribuição de BI-RADS e densidade por dataset e por *split* | S4 |
| T3 | Ablação de pré-processamento `B0`–`B4`: mAP@0.5 massa / calcificação, com IC 95% | S7 |
| T4 | Detecção: sensibilidade @ 0,25 / 0,5 / 1,0 FP por imagem + mAP@0.5, por categoria de achado | S6 |
| T5 | Comparação com a literatura no VinDr (Mammo-CLIP 0,58; Mammo-FM 0,58; YOLOv12-L 0,59; MM-DETR 0,654) × seu resultado | S8 |
| T6 | Classificação por mama: AUROC, macro-F1, kappa quadrático; Mammo-CLIP EN-B2 × EfficientNet-B2 ImageNet | S8 |
| T7 | Extração de laudo: precisão/*recall*/F1 por campo (`categoria`, `lado`, `região`, cada descritor, BI-RADS), `E0` × `E1` × `E2` | S11 |
| T8 | Concordância inter-anotador (kappa de Cohen) nos 50 laudos duplicados | S10 |
| T9 | Auditoria: precisão e *recall* por tipo de alerta `A1`–`A8`, × baselines `Z0`/`Z1`/`Z2` | S14 |
| T10 | Sensibilidade estratificada por severidade (1–5) | S14 |
| T11 | Teste externo: treino VinDr → teste CDD-CESM e DMID (queda esperada; declare como robustez) | S14 |
| T12 | Análise de erro: 60 casos categorizados por causa raiz | S15 |
| T13 | Plausibilidade da injeção sintética: taxa + kappa entre 2 avaliadores | S13 |
| T14 | Checklist **CLAIM 2024** preenchido (44 itens, Yes/No/NA justificado) | S16 — anexo |

---

## 13. Referências essenciais (por onde começar a ler)

**Leia estas cinco primeiro, nesta ordem:**
1. **Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports** — MICCAI 2025, [arXiv:2509.21356](https://arxiv.org/abs/2509.21356). O trabalho mais próximo do seu; é dele que sai a lacuna de omissão.
2. **CorBenchX** — [arXiv:2505.12057](https://arxiv.org/pdf/2505.12057). A tarefa formalmente idêntica à sua, em tórax; taxonomia com omissão e a barra de 50,6%.
3. **Mammo-CLIP** — MICCAI 2024, [arXiv:2405.12255](https://arxiv.org/abs/2405.12255) · [código](https://github.com/batmanlab/Mammo-CLIP). Seu *backbone*, e o Mammo-FActOR é o embrião do casamento sentença↔região.
4. **VinDr-Mammo** — *Scientific Data* 2023, [artigo](https://www.nature.com/articles/s41597-023-02100-7) · [código](https://github.com/vinbigdata-medical/vindr-mammo). Seu dataset de treino; leia a Tabela 5 e a seção de anotação.
5. **CLAIM 2024** — *Radiology: AI*, DOI [10.1148/ryai.240300](https://pubs.rsna.org/doi/10.1148/ryai.240300). O checklist que vai estruturar sua metodologia.

**Datasets**
- VinDr-Mammo — [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) · [Scientific Data](https://www.nature.com/articles/s41597-023-02100-7)
- CBIS-DDSM — [TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) · [Lee et al., Sci Data 2017](https://www.nature.com/articles/sdata2017177)
- INbreast — [Moreira et al., 2012 (PDF)](https://www.cse.unr.edu/~bebis/CS791/Fall2025/Papers/DataSets/2012%20-%20INBreast%20Toward%20a%20Full-field%20Digital%20Mammographic%20Database.pdf)
- CDD-CESM — [TCIA](https://www.cancerimagingarchive.net/collection/cdd-cesm/) · [Khaled et al., Sci Data 2022](https://www.nature.com/articles/s41597-022-01238-0) · [planilha de anotações](https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx)
- DMID — [figshare, CC BY 4.0](https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883)
- Laudos em espanhol (Paraguai), 4.357 laudos — Zenodo [10.5281/zenodo.14827680](https://doi.org/10.5281/zenodo.14827680)
- Revisão FAIR dos datasets de mamografia — [Sci Data 2023](https://www.nature.com/articles/s41597-023-02430-6)

**Detecção e classificação em mamografia**
- Mammo-FM — [arXiv:2512.00198](https://arxiv.org/pdf/2512.00198)
- MaMA — [arXiv:2409.18119](https://arxiv.org/abs/2409.18119)
- MM-DETR (multi-vista) — [Deep-Breath @ MICCAI 2025](https://link.springer.com/chapter/10.1007/978-3-032-05559-0_26)
- BRAIxDet — [Medical Image Analysis 2024](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001178) · [arXiv:2301.13418](https://arxiv.org/html/2301.13418v4)
- M&M (detector esparso) — [MICCAI 2023](https://arxiv.org/abs/2308.06420)
- Comparação YOLO × RT-DETR — [Cancers 18(1):70](https://doi.org/10.3390/cancers18010070)
- Ablação de pré-processamento + YOLOv12 — [J. Imaging 11(9):314](https://www.mdpi.com/2313-433X/11/9/314)
- Perda ordinal para BI-RADS — [J. Clin. Med. 15(1):365](https://www.mdpi.com/2077-0383/15/1/365)
- Densidade e multi-dataset — [Mathematics 14(12):2080](https://www.mdpi.com/2227-7390/14/12/2080)
- Generalização entre 4 países — [Sensors 26(12):3911](https://www.mdpi.com/1424-8220/26/12/3911)
- Desafio RSNA 2023, resultados — [Radiology](https://pubs.rsna.org/doi/10.1148/radiol.241447) · [código do 1º lugar](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)
- Alta resolução por janelas locais — [Frontiers in Oncology 2025](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1601929/full)
- Resolução e AUC — [arXiv:2503.19945](https://arxiv.org/html/2503.19945v3) · [Bioengineering 10(5):534](https://www.mdpi.com/2306-5354/10/5/534)
- Recorte da mama (ablação MamT⁴) — [arXiv:2411.01669](https://arxiv.org/html/2411.01669)
- Pré-processamento, comparação sistemática — [Sci Rep 2025](https://www.nature.com/articles/s41598-025-30724-9) · [Technologies 14(4):236](https://www.mdpi.com/2227-7080/14/4/236)
- CLAHE, evidência a favor — [Cancers 16(2):322](https://doi.org/10.3390/cancers16020322); contra — [Algorithms 18(12):796](https://www.mdpi.com/1999-4893/18/12/796)
- Codebase transparente CBIS-DDSM — [Sci Rep 2024](https://www.nature.com/articles/s41598-024-78648-0)

**Laudos, extração e auditoria**
- LLMs de pesos abertos × GPT-4o em extração — [Radiology](https://pubs.rsna.org/doi/10.1148/radiol.240895)
- Qwen-0.5B superando CheXpert — [JMIR Med Inform 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11970564/)
- Negação em laudos (medspaCy × BERT) — [J Imaging Inform Med 2024](https://link.springer.com/article/10.1007/s10278-024-01274-9)
- BI-RADS por LLM: AC1 0,49–0,57 — [Radiology 2024](https://pubs.rsna.org/doi/full/10.1148/radiol.232133)
- TECRR (5.046 laudos, BI-RADS) — [BMC Med Inform Decis Mak 2024](https://link.springer.com/article/10.1186/s12911-024-02717-7)
- RadGraph-XL — [Findings ACL 2024](https://aclanthology.org/2024.findings-acl.765/)
- GREEN — [arXiv:2405.03595](https://arxiv.org/html/2405.03595v1) · RadFact — [código](https://github.com/microsoft/RadFact) · FineRadScore — [arXiv:2405.20613](https://arxiv.org/html/2405.20613v2) · RaTEScore — [EMNLP 2024](https://angelakeke.github.io/RaTEScore/)
- ReXErr (12 tipos de erro) — [arXiv:2409.10829](https://arxiv.org/html/2409.10829v1) · [ReXErr-v1, PhysioNet](https://physionet.org/content/rexerr-v1/1.0.0/)
- RADAR (revisão de laudo ancorada em imagem) — [arXiv:2603.06681](https://arxiv.org/html/2603.06681)
- RadFlag — [arXiv:2411.00299](https://arxiv.org/html/2411.00299v1) · ReXTrust — [arXiv:2412.15264](https://arxiv.org/html/2412.15264v3)
- MammoVQA (VLMs ≈ palpite aleatório) — [Nature Communications 2025](https://www.nature.com/articles/s41467-025-66507-z)
- LLaVA-Rad + CheXprompt (taxonomia de 6 erros) — [Nature Communications 2025](https://www.nature.com/articles/s41467-025-58344-x)
- AMRG (MedGemma-4B + LoRA em DMID) — [arXiv:2508.09225](https://arxiv.org/html/2508.09225v1) · MammoWise (RAG local) — [arXiv:2602.22462](https://arxiv.org/html/2602.22462)
- VisualCheXbert (discordância laudo↔imagem é sistemática) — [ACM CHIL 2021](https://dl.acm.org/doi/pdf/10.1145/3450439.3451862)
- Dupla leitura assistida por IA em tórax — [European Radiology 2024](https://link.springer.com/article/10.1007/s00330-024-10676-w)
- Concordância radiologia↔patologia em mama (só 13 discordâncias em 766) — [Diagnostics 16(9):1249](https://www.mdpi.com/2075-4418/16/9/1249)
- *Generate to Ground* (condicionamento textual dobra o mIoU) — [MIDL 2025, arXiv:2507.12236](https://arxiv.org/html/2507.12236v1)
- Qualidade informacional de laudos brasileiros (22.247 laudos) — [Radiologia Brasileira 2010](https://www.scielo.br/j/rb/a/8jbwGc9KhMvBXHBxtgyW4Fz/?lang=pt)
- BioBERTpt — [ClinicalNLP@EMNLP 2020](https://aclanthology.org/2020.clinicalnlp-1.7/) · BRAX — [Sci Data 2022](https://www.nature.com/articles/s41597-022-01608-8)
- 4º Desafio de IA da SPR (laudos de mamografia em PT-BR) — [SPR](https://noticias.spr.org.br/jpr-2026/4o-desafio-de-ia-explora-interpretacao-de-laudos-de-mamografia)

**Métricas, estatística e relato**
- *Metrics Reloaded* — [Nature Methods 21:195–212, 2024](https://www.nature.com/articles/s41592-023-02151-z)
- CLAIM 2024 — [Radiology: AI](https://pubs.rsna.org/doi/10.1148/ryai.240300)
- DeLong et al., *Biometrics* 44(3):837–845, 1988 · Landis & Koch, *Biometrics* 33(1):159–174, 1977
- `RJafroc` (wAFROC/MRMC, se comparar contra leitores humanos) — [CRAN](https://cran.r-project.org/package=RJafroc)
- Brady, *Error and discrepancy in radiology: inevitable or avoidable?* — [Insights into Imaging 8:171–182, 2017](https://doi.org/10.1007/s13244-016-0534-1) — **use na introdução**

---

## 14. As cinco coisas para fazer nesta semana

1. Baixar `Radiology-manual-annotations.xlsx` do CDD-CESM — [link direto](https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx). Sem essa planilha não há padrão de referência de descritores. **20 segundos de download.**
2. Baixar `Reports.zip` + `Metadata.xlsx` do DMID (figshare) — 0,2 MB + 40 KB. Dá comparabilidade com um baseline publicado.
3. Abrir 10 laudos do INbreast com `encoding='latin-1'` e 10 `.docx` do CDD-CESM e **anotar à mão**, num papel, quais campos do `FindingRecord` cada um preenche. Isso vai mudar seu esquema, e é melhor que mude agora.
4. Rodar o `build_manifest.py` do repositório sobre 200 DICOMs do VinDr e olhar a tabela de metadados. Você vai descobrir quantas imagens são `FOR PROCESSING` (descartáveis) e qual fabricante domina.
5. Mostrar ao orientador as §0.1 (as seis correções) e §10 (o corte de escopo pré-aprovado). São as duas conversas que precisam acontecer **antes** de você escrever código de treino.

---

*Documento gerado com revisão de literatura verificada em fontes primárias (agosto/2026). Números marcados como verificados foram lidos nas fontes citadas ou medidos diretamente nos seus arquivos em `D:\Erick\Dataset`. Onde a literatura não permitiu verificação, o texto diz isso explicitamente.*
