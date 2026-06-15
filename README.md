# Sumarizacao de Abstracts Biomedicos com BioBART (PubMed)

Projeto final da disciplina **Topicos em Processamento de Linguagem Natural** (IESB).
Fine-tuning do modelo **BioBART** para **sumarizacao abstrativa** de artigos cientificos
biomedicos, usando o dataset [`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization)
e avaliacao por **ROUGE**.

## 🔗 Links do produto

| Item | Link |
|---|---|
| 🌐 Aplicacao web (Gradio / HF Spaces) | `https://huggingface.co/spaces/SEU_USUARIO/biobart-pubmed-sumarizacao` |
| 🧬 Modelo treinado (HF Hub) | `https://huggingface.co/SEU_USUARIO/biobart-pubmed-summarization` |
| 📄 Relatorio final | [`relatorio/relatorio_final.md`](relatorio/relatorio_final.md) |

## Arquitetura

```
Treino (Colab GPU)          Publicacao            Inferencia (HF Spaces)
ccdv/pubmed-summarization  ─►  HuggingFace Hub  ─►  App Gradio (UI + API REST)
   + BioBART fine-tuning        (modelo)              usuario cola artigo → resumo
   + avaliacao ROUGE
```

Treino **offline** no Google Colab (GPU T4) e inferencia **online** no HuggingFace Spaces,
conectados pelo Hub. Detalhes no relatorio.

## Estrutura do repositorio

```
.
├── eda_pubmed.ipynb                    # Analise exploratoria dos dados (artefato parcial)
├── train/
│   └── finetune_biobart_pubmed.ipynb   # Pipeline de fine-tuning (rodar no Colab)
├── app/
│   ├── app.py                          # Aplicacao Gradio (HuggingFace Space)
│   ├── requirements.txt
│   └── README.md                       # Config do Space (cabecalho YAML)
├── relatorio/
│   └── relatorio_final.md              # Relatorio final (projeto + arquitetura + resultados)
├── plano do projeto.txt                # Plano (artefato parcial)
├── requirements.txt                    # Dependencias para reproduzir localmente
└── README.md
```

## Stack

HuggingFace `transformers` · `datasets` · `evaluate` (ROUGE) · PyTorch · Gradio · HuggingFace Spaces.

## Como reproduzir

**1. Treino (Google Colab):** abra `train/finetune_biobart_pubmed.ipynb`, selecione runtime com
GPU (T4), execute as celulas. Ele faz login no Hub, treina, avalia (fine-tuned vs baseline) e
publica o modelo. Anote o `MODEL_ID` resultante.

**2. Aplicacao (local):**
```bash
pip install -r app/requirements.txt
# edite MODEL_ID em app/app.py (ou defina a env var MODEL_ID)
python app/app.py
```

**3. Deploy (HuggingFace Spaces):** crie um Space (SDK = Gradio) e envie o conteudo de `app/`.
