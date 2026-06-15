"""
App Gradio - Sumarizacao de Abstracts Biomedicos com BioBART (fine-tuned em PubMed).

Carrega o modelo publicado no HuggingFace Hub e expoe uma interface web + API REST
(em /api/predict). Pensado para rodar no HuggingFace Spaces (CPU gratuito).

Como apontar para o seu modelo:
  - edite a constante MODEL_ID abaixo, OU
  - defina a variavel de ambiente MODEL_ID (em Spaces: Settings > Variables).
Para testar o app ANTES do fine-tuning terminar, use o modelo base:
  MODEL_ID = "GanjinZero/biobart-v2-base"
"""

import os
import gradio as gr
from transformers import pipeline

# >>> TROQUE pelo id do SEU modelo no Hub apos o treino (ex.: "maycon/biobart-pubmed-summarization")
MODEL_ID = os.environ.get("MODEL_ID", "SEU_USUARIO/biobart-pubmed-summarization")

MAX_INPUT_TOKENS = 1024  # mesma janela usada no fine-tuning

print(f"Carregando modelo: {MODEL_ID} ...")
summarizer = pipeline("summarization", model=MODEL_ID, tokenizer=MODEL_ID)
print("Modelo carregado.")


def summarize(article: str, max_length: int = 256, min_length: int = 40, num_beams: int = 4) -> str:
    article = (article or "").strip()
    if not article:
        return "Cole o texto de um artigo/abstract biomedico para gerar o resumo."
    out = summarizer(
        article,
        truncation=True,
        max_length=int(max_length),
        min_length=int(min_length),
        num_beams=int(num_beams),
    )
    return out[0]["summary_text"]


EXAMPLE_ARTICLE = (
    "atrial fibrillation ( af ) is the most common sustained arrhythmia in western countries , "
    "with an estimated 30 million patients affected by 2050 across united states and europe alone . "
    "atrial fibrillation has a significant impact on morbidity mainly related to symptoms , heart "
    "failure , and thromboembolic events and is the most frequent arrhythmic cause of hospital "
    "admission . to date , the most effective treatment for af is radiofrequency catheter ablation , "
    "and pulmonary vein antrum isolation ( pvai ) is the mainstay of such an approach . the major "
    "drawback of catheter ablation of af consists in its potential risk of periprocedural "
    "complications , such as thromboembolic events and bleeding , which makes a correct management "
    "of anticoagulation essential to prevent such complications ."
)

DESCRIPTION = """
# 🧬 Sumarizacao de Abstracts Biomedicos — BioBART + PubMed

Modelo **BioBART** ajustado (*fine-tuning*) no dataset **PubMed Summarization** para gerar um
**resumo conciso** a partir do corpo de um artigo cientifico biomedico.

Projeto final da disciplina *Topicos em Processamento de Linguagem Natural*.
Cole um texto em ingles e clique em **Sumarizar**.
"""

with gr.Blocks(title="BioBART PubMed Summarization") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(
                label="Artigo / Abstract (ingles)",
                placeholder="Cole aqui o texto do artigo biomedico...",
                lines=14,
            )
        with gr.Column(scale=1):
            max_len = gr.Slider(64, 400, value=256, step=8, label="Tamanho maximo do resumo")
            min_len = gr.Slider(10, 120, value=40, step=5, label="Tamanho minimo do resumo")
            beams = gr.Slider(1, 8, value=4, step=1, label="Num. de beams (qualidade x tempo)")
            btn = gr.Button("Sumarizar", variant="primary")
    out = gr.Textbox(label="Resumo gerado", lines=8)

    btn.click(summarize, inputs=[inp, max_len, min_len, beams], outputs=out, api_name="predict")
    gr.Examples(examples=[[EXAMPLE_ARTICLE]], inputs=[inp])

    gr.Markdown(
        f"Modelo: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) · "
        "Base: `GanjinZero/biobart-v2-base` · Dataset: `ccdv/pubmed-summarization`"
    )

if __name__ == "__main__":
    demo.launch()
