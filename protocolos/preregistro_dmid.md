# Pré-registro: validação da calibração local na DMID

Registrado em 03/10/2026, **antes** do download das imagens da DMID e antes de
qualquer predição nela. Este arquivo e `configs/dmid_protocol.yaml` vão para o git
num commit próprio; o hash desse commit é a prova de que o protocolo veio antes dos
resultados. Nada aqui pode mudar depois de rodar os modelos na DMID. Qualquer desvio
é reportado à parte, com o motivo.

## Por que

No INbreast, os limiares calibrados no VinDr-Mammo não se transferiram (acurácia
balanceada de 0,473, pior que não alertar). A recalibração no próprio INbreast
recuperou 0,713, mas foi decidida depois de ver o resultado. A pergunta deste
protocolo é: **calibrar os limiares com exames consistentes do próprio serviço
funciona numa base que não foi usada para nada até agora?**

## O que já foi visto da DMID antes do registro

Só a estrutura dos arquivos pequenos (`Metadata.xlsx` e `Reports.zip`):

- 510 imagens, um laudo em inglês por imagem, campo `BIRADS:` com uma ou mais categorias.
- Pela maior categoria escrita: 359 imagens de categoria 1 a 3, 150 de 4 ou 5, 1 de categoria 0.
- A planilha traz a vista (CC/MLO, direita/esquerda), o tipo de tecido e as lesões,
  mas **não traz a identificação da paciente**.

Nenhuma imagem foi aberta e nenhum modelo foi rodado na DMID.
O extrator por regras foi rodado em dois laudos de exemplo só para confirmar que lê o
inglês da DMID; nada foi ajustado. Ele já mostrou um ponto fraco previsível: nos laudos
da DMID cada achado fica numa linha sem ponto final, e a negação de uma linha pode
alcançar a seguinte. O extrator segue congelado assim mesmo.

## Base

DMID (Digital mammography Dataset for Breast Cancer Diagnosis Research), figshare,
DOI 10.6084/m9.figshare.24522883.v2, licença CC BY 4.0. Arquivos usados:
`DICOM Images.zip`, `Reports.zip`, `Metadata.xlsx`.

## Unidade, categoria e grupo

- **Unidade:** a imagem, porque cada imagem tem o seu laudo.
- **Categoria do laudo:** a maior categoria escrita no campo `BIRADS:` (até o primeiro
  ponto). 4A, 4B e 4C contam como 4. Entram as categorias 1 a 5; a categoria 0, ou um
  laudo sem categoria, sai.
- **Lado da fronteira de biópsia:** 1 a 3 contra 4 e 5, pela categoria do próprio laudo
  (a DMID não tem outra referência de categoria).
- **Grupo (para dividir e para os pares trocados):** a paciente, pelo `PatientID` do DICOM,
  se ele existir e tiver mais de um valor; senão o `StudyInstanceUID`; senão a própria
  imagem. A regra usada e o número de grupos são reportados.

## Divisão

Por grupo, metade para **calibração** e metade para **avaliação**, estratificada pelo
lado da fronteira (grupo com alguma imagem 4 ou 5 conta como lado 4 e 5), com semente
20260819. A divisão é feita e congelada (hash no git) antes de qualquer predição.

## Calibração (metade de calibração)

A mesma regra da calibração no VinDr-Mammo, com alfa de 5%, usando só os pares
originais (cada imagem com o seu laudo):

- massa e microcalcificação: percentil 97,5 da pontuação nas imagens de categoria 1 e 2;
- classificador: percentil 95 de P(4)+P(5) nas imagens de categoria 1 a 3.

## Avaliação (metade de avaliação, uma vez só)

- Pares originais: cada imagem com o seu laudo.
- Pares trocados: cada imagem com até 5 laudos de outras pacientes da metade de
  avaliação, do lado oposto da fronteira, sorteados com a semente 20260819.
- Regras V1 a V4 iguais às do TCC; extrator por regras congelado (léxico em inglês sem
  nenhum ajuste para a DMID); modelos congelados.

## Desfechos

- **Principal:** acurácia balanceada do sistema principal (V1+V3+V4) com os limiares
  calibrados na DMID, na metade de avaliação, com IC 95% por bootstrap de grupos.
- **Critério de sucesso, fixado agora:** limite inferior do IC 95% acima de 0,5 (melhor
  que nunca alertar).
- **Secundários:** especificidade, trocas rebaixadas e elevadas encontradas; o mesmo com
  os limiares do VinDr (teste de transferência); principal com V2; só imagem, só
  detectores, só classificador e só texto; McNemar do principal contra só classificador
  e contra só texto.

## Tamanho do pixel

Usa o tamanho do DICOM. Se o DICOM não informar, a microcalcificação usa o tamanho
informado na publicação da base; se nenhum estiver documentado, a imagem não é
reamostrada e isso é declarado.

## O que não pode acontecer

Retreinar modelo, mudar regra, mudar o léxico, escolher outro alfa ou outra divisão
depois de ver qualquer predição na DMID.
