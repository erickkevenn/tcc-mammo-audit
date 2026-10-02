# Texto do TCC (LaTeX, abnTeX2)

## Como compilar

**Overleaf (mais simples):** compacte a pasta `texto/` em um .zip, crie um projeto com "Upload Project" e compile. O compilador padrão (pdfLaTeX) funciona.

**No Windows (MiKTeX):** com o MiKTeX instalado (ele baixa os pacotes que faltarem), abra o terminal nesta pasta e rode:

```
latexmk -pdf main.tex
```

Para limpar os arquivos temporários: `latexmk -c`.

## Estrutura

| Arquivo | Conteúdo | Situação |
|---|---|---|
| `main.tex` | Capa, folha de rosto, listas, ordem dos capítulos | Faltam curso, grau e orientador |
| `capitulos/00-resumo.tex` | Resumo e abstract | Resumo provisório; abstract pendente |
| `capitulos/01-introducao.tex` | Contexto, problema, objetivos, escopo | Rascunho completo |
| `capitulos/02-fundamentacao.tex` | BI-RADS, detecção, FROC, trabalhos relacionados | Rascunho; faltam algumas referências |
| `capitulos/03-metodologia.tex` | Dados, divisão, pré-processamento, modelos, auditor, avaliação | Parte de imagem completa; extrator E1 e anotação pendentes |
| `capitulos/04-resultados.tex` | Pré-processamento, detector, ablação | Detector e ablação prontos; classificador, extração, auditoria e teste pendentes |
| `capitulos/05-discussao.tex` | Comparação com a literatura, ablação, limitações | Rascunho da parte de imagem |
| `capitulos/06-conclusao.tex` | Conclusão e trabalhos futuros | Parcial |
| `referencias.bib` | Referências (conferidas na fonte) | Em crescimento |
| `figuras/` | Figuras do texto (PNG/JPG/PDF) | Vazia |

## Convenção

Tudo o que falta aparece em **vermelho** no PDF, com o comando `\pendente{...}`. Para listar o que falta:

```
findstr /s /n "\pendente" *.tex
```

## Regras do texto

- Sem travessões no corpo do texto; intervalos escritos como "0,77 a 0,89".
- Números de resultado só da validação, até a avaliação final no teste (feita uma única vez).
- Não usar "validado por especialista", "revisão clínica" ou "aprovado por radiologista". Usar "anotadores treinados no protocolo de anotação" e "referência derivada das anotações públicas do VinDr-Mammo".
- Toda referência nova deve ser conferida na fonte antes de entrar no `referencias.bib`.
- Não reproduzir laudos do INbreast na íntegra.
