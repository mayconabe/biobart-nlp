"""
App Gradio - Sumarizacao de Abstracts Biomedicos com BioBART (fine-tuned em PubMed).

Carrega o modelo publicado no HuggingFace Hub e expoe uma interface web + API REST
(em /api/predict). Pensado para rodar no HuggingFace Spaces (CPU gratuito).

Usa AutoModelForSeq2SeqLM + model.generate() diretamente (sem pipeline), o que e
robusto entre versoes do transformers (o pipeline 'summarization' foi removido na v5).

Como apontar para o seu modelo:
  - edite a constante MODEL_ID abaixo, OU
  - defina a variavel de ambiente MODEL_ID (em Spaces: Settings > Variables).
Para testar com o modelo base, use:
  MODEL_ID = "GanjinZero/biobart-v2-base"
"""

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Modelo fine-tuned publicado no HuggingFace Hub (pode ser sobrescrito pela env var MODEL_ID)
MODEL_ID = os.environ.get("MODEL_ID", "mayconabe/biobart-pubmed-summarization")

MAX_INPUT_TOKENS = 1024  # mesma janela usada no fine-tuning

print(f"Carregando modelo: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("Modelo carregado.")


@torch.no_grad()
def summarize(article: str, max_length: int = 256, min_length: int = 40, num_beams: int = 4) -> str:
    article = (article or "").strip()
    if not article:
        return "Cole o texto de um artigo/abstract biomedico para gerar o resumo."
    inputs = tokenizer(
        article,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    output_ids = model.generate(
        **inputs,
        max_length=int(max_length),
        min_length=int(min_length),
        num_beams=int(num_beams),
        no_repeat_ngram_size=3,  # evita repetir trigramas (reduz copia literal/loops)
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


EXAMPLE_ARTICLE = (
    "type 2 diabetes mellitus ( t2dm ) is a chronic metabolic disorder characterized by insulin "
    "resistance and progressive beta cell dysfunction , affecting more than 460 million people "
    "worldwide and projected to rise to 700 million by 2045 . the disease is associated with serious "
    "microvascular and macrovascular complications , including nephropathy , retinopathy , neuropathy "
    ", and cardiovascular disease , which substantially increase morbidity , mortality , and health "
    "care costs . despite the availability of several classes of glucose lowering agents , a large "
    "proportion of patients fail to achieve adequate glycemic control , highlighting the need for "
    "novel therapeutic strategies . metformin remains the first line pharmacological therapy for most "
    "patients with t2dm because of its efficacy , low cost , and favorable safety profile . however , "
    "its effect on long term cardiovascular outcomes when combined with newer agents such as sodium "
    "glucose cotransporter 2 ( sglt2 ) inhibitors has not been fully characterized in real world "
    "populations . in this study , we conducted a retrospective cohort analysis of 3,482 adult "
    "patients with type 2 diabetes treated at a tertiary care center between 2015 and 2021 . patients "
    "were classified into two groups according to their treatment regimen : metformin monotherapy "
    "( n = 1,914 ) and metformin combined with an sglt2 inhibitor ( n = 1,568 ) . the primary outcome "
    "was a composite of major adverse cardiovascular events , defined as nonfatal myocardial "
    "infarction , nonfatal stroke , or cardiovascular death . secondary outcomes included changes in "
    "glycated hemoglobin ( hba1c ) , body weight , and estimated glomerular filtration rate . cox "
    "proportional hazards models were used to estimate adjusted hazard ratios , controlling for age , "
    "sex , baseline hba1c , duration of diabetes , and established cardiovascular disease . over a "
    "median follow up of 3.6 years , the combination therapy group experienced a significantly lower "
    "rate of major adverse cardiovascular events compared with the monotherapy group ( 8.1% versus "
    "12.4% ; adjusted hazard ratio 0.71 , 95% confidence interval 0.59 to 0.86 ; p < 0.001 ) . "
    "patients receiving the sglt2 inhibitor also showed greater reductions in hba1c and body weight , "
    "as well as a slower decline in renal function , and the incidence of hospitalization for heart "
    "failure was reduced by approximately one third . adverse events were generally mild ; genital "
    "mycotic infections were more frequent in the combination group , but rates of diabetic "
    "ketoacidosis and severe hypoglycemia were low . these findings suggest that the early addition "
    "of an sglt2 inhibitor to metformin is associated with meaningful cardiovascular and renal "
    "benefits in a real world population of patients with type 2 diabetes , supporting individualized "
    "treatment intensification ."
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
