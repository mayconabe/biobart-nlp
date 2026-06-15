# Relatorio Final — Sumarizacao Automatica de Abstracts Biomedicos com BioBART

**Disciplina:** Topicos em Processamento de Linguagem Natural — Prof. Alexandre Roriz
**Aluno:** Maycon Machado
**Data:** 16/06/2026

> **Links do produto** (preencher antes de entregar):
> - 🌐 Aplicacao (HuggingFace Space): `https://huggingface.co/spaces/mayconabe/biobart-pubmed-sumarizacao`
> - 💻 Repositorio (GitHub): `https://github.com/mayconabe/nlp-biobart-pubmed`
> - 🧬 Modelo (HuggingFace Hub): `https://huggingface.co/mayconabe/biobart-pubmed-summarization`

---

## 1. Introducao e motivacao

O volume de publicacoes cientificas na area biomedica cresce em ritmo que torna inviavel a
leitura manual exaustiva por pesquisadores e profissionais de saude. Ferramentas de
**sumarizacao automatica** ajudam na triagem dessa literatura, gerando resumos concisos a partir
de textos tecnicos extensos.

Este projeto realiza o **fine-tuning do modelo BioBART** — uma variante do BART pre-treinada em
literatura biomedica — para a tarefa de **sumarizacao abstrativa** de artigos cientificos,
usando o dataset publico PubMed Summarization. O objetivo e demonstrar que um modelo de linguagem
pre-treinado em dominio especifico pode ser ajustado de forma eficiente para superar a sua propria
baseline (sem fine-tuning) nessa tarefa.

## 2. Tarefa de PLN e dataset

- **Tarefa:** sumarizacao abstrativa (sequence-to-sequence). Entrada = corpo do artigo;
  saida = resumo conciso e fiel ao conteudo.
- **Dataset:** [`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization)
  — ~119.9k exemplos de treino, 6.6k de validacao e 6.7k de teste. Cada exemplo tem os campos
  `article` (texto completo) e `abstract` (resumo de referencia).
- **Modelo base:** [`GanjinZero/biobart-v2-base`](https://huggingface.co/GanjinZero/biobart-v2-base)
  (janela de 1024 tokens, vocabulario de ~85k).
- **Metrica:** ROUGE-1, ROUGE-2 e ROUGE-L, padrao para sumarizacao.

## 3. Analise exploratoria (resumo)

A EDA completa esta em `eda_pubmed.ipynb`. Principais achados sobre uma amostra de 5.000 exemplos
de treino (tokenizados com o tokenizer do BioBART):

| Metrica | Article (tokens) | Abstract (tokens) |
|---|---|---|
| Media | 4.325 | 281 |
| Mediana | 3.508 | 289 |
| P95 | 10.063 | 461 |
| Maximo | 111.658 | 602 |

- Os **artigos sao muito longos** (mediana ~3.5k tokens), muito acima da janela de 1024 do BART.
- Os **abstracts sao bem comportados** (P99 < 512 tokens).
- **Razao de compressao** mediana ~13x (em tokens) — tarefa de sumarizacao "pesada", nao parafrase.
- **Truncamento:** com `max_input_length=1024`, ~95% dos artigos sao truncados; com
  `max_target_length=256`, < 1% dos abstracts sao afetados.

**Decisao de pre-processamento (config "moderada" da EDA):** `max_input_length = 1024`,
`max_target_length = 256`. Filtros de qualidade aplicados ao treino: remover pares com abstract
< 20 palavras e com razao de compressao degenerada (< 2x ou > 100x).

## 4. Metodologia / configuracao do fine-tuning

- Subconjunto de treino: **8.000** exemplos (limitado pelo prazo; val/teste preservados).
- Tokenizacao: artigo truncado em 1024 tokens → alvo (abstract) em 256 tokens.
- Treinador: `Seq2SeqTrainer` (HuggingFace Transformers), `predict_with_generate=True`.
- Hiperparametros: learning rate `5e-5`, batch efetivo 8 (4 × grad-accum 2), 2 epocas,
  `weight_decay=0.01`, `warmup_ratio=0.05`, `fp16` (GPU), beam search com 4 beams na geracao.
- Ambiente: **Google Colab (GPU NVIDIA T4)** — o treino local foi inviabilizado pela GPU
  disponivel (MX150, 2 GB).
- Avaliacao: ROUGE no subconjunto de teste (n=500), comparando **fine-tuned vs baseline**
  (`biobart-v2-base` sem ajuste) sob as mesmas condicoes de geracao.

## 5. Arquitetura implementada

O sistema separa **treino offline** (Colab) de **inferencia online** (HuggingFace Spaces),
conectados pelo HuggingFace Hub, onde o modelo treinado e publicado.

```
┌─ TREINO (offline · Google Colab · GPU T4) ──────────────────────────────┐
│  ccdv/pubmed-summarization  (HF Datasets, ~120k train)                   │
│        │  subconjunto + pre-processamento (tokenize 1024/256 + filtros)  │
│        ▼                                                                  │
│  GanjinZero/biobart-v2-base ──fine-tuning (Seq2SeqTrainer, fp16)──► ckpt │
│        │  avaliacao ROUGE-1/2/L  (fine-tuned  vs  baseline)              │
│        ▼                                                                  │
│  trainer.push_to_hub() ──► HuggingFace Hub: <user>/biobart-pubmed-summ   │
└──────────────────────────────────────────────────────────────────────────┘
                                  │  (modelo publicado)
                                  ▼
┌─ INFERENCIA (online · HuggingFace Spaces · CPU) ────────────────────────┐
│  Usuario (navegador) ──cola abstract/artigo──► App Gradio (app.py)       │
│        │  pipeline("summarization") carrega o modelo do Hub             │
│        │  generate (beam search, max_len=256)                            │
│        ▼                                                                  │
│  Resumo gerado ──► exibido na UI  (+ endpoint REST /api/predict)         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Componentes e tecnologias:**
- *Dados/modelagem:* HuggingFace `datasets`, `transformers`, `evaluate` (ROUGE), PyTorch.
- *Publicacao do modelo:* HuggingFace Hub.
- *Aplicacao web:* Gradio (UI + API REST automatica), hospedada no HuggingFace Spaces.
- *Codigo-fonte:* GitHub.

## 6. Resultados

### 6.1 Quantitativo (ROUGE — subconjunto de teste, n=500)

> Preencher com os numeros impressos pela celula de comparacao do notebook de treino.

| Modelo | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| Baseline (`biobart-v2-base`, sem fine-tuning) | _.._ | _.._ | _.._ | _.._ |
| **BioBART fine-tuned (este trabalho)** | **_.._** | **_.._** | **_.._** | **_.._** |

Observacao esperada: o modelo ajustado deve superar a baseline em todas as metricas ROUGE,
confirmando o ganho do fine-tuning em dominio especifico.

### 6.2 Qualitativo

> Colar 1–2 exemplos (artigo → resumo gerado × abstract de referencia) da secao 8 do notebook.

## 7. Limitacoes e trabalhos futuros

- **Truncamento de artigos longos:** a janela de 1024 tokens descarta grande parte dos artigos.
  Estrategias futuras: janela deslizante, selecao de secoes-chave (introducao + conclusao) ou
  modelos de contexto longo (Longformer/LED).
- **Subconjunto de treino** reduzido por restricao de tempo/GPU; treinar com mais dados tende a
  melhorar as metricas.
- **Cold start** no Space: a primeira requisicao baixa o modelo e e mais lenta (inferencia em CPU).

## 8. Conclusao

O projeto entregou, de ponta a ponta, um pipeline de sumarizacao biomedica: da analise dos dados
ao fine-tuning do BioBART, avaliacao por ROUGE contra a baseline e disponibilizacao de uma
aplicacao web publica que demonstra o modelo funcionando. A arquitetura desacoplada
(treino no Colab → modelo no Hub → app no Spaces) e reprodutivel e usa apenas recursos gratuitos.
