---
title: BioBART PubMed Summarization
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Sumarizacao de Abstracts Biomedicos — BioBART + PubMed

Demo (Gradio) do modelo **BioBART** ajustado por *fine-tuning* no dataset
[`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization)
para sumarizacao de artigos cientificos biomedicos.

- **Base:** `GanjinZero/biobart-v2-base`
- **Tarefa:** sumarizacao abstrativa (artigo → resumo conciso)
- **Avaliacao:** ROUGE-1/2/L (fine-tuned vs baseline)

## Configuracao

O id do modelo carregado vem da constante `MODEL_ID` em `app.py` ou da variavel de
ambiente `MODEL_ID` (em Settings > Variables, no Space). Aponte para o seu modelo
publicado no Hub, por exemplo `SEU_USUARIO/biobart-pubmed-summarization`.

## Rodar localmente

```bash
pip install -r requirements.txt
python app.py
```

A interface tambem expoe um endpoint REST automatico em `/api/predict`.
