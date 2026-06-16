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


# ----------------------------------------------------------------------------------
# Artigos de exemplo (longos, estilo PubMed) para demonstrar a sumarizacao.
# ----------------------------------------------------------------------------------
EXAMPLE_DIABETES = (
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

EXAMPLE_NSCLC = (
    "lung cancer remains the leading cause of cancer related death worldwide , and non - small cell "
    "lung cancer ( nsclc ) accounts for approximately 85% of all cases . the majority of patients are "
    "diagnosed at an advanced stage , when curative surgical resection is no longer possible , and "
    "historically the prognosis for metastatic disease has been poor , with five year survival rates "
    "below 5% . platinum based chemotherapy was for decades the standard of care , but its benefit "
    "reached a plateau and was accompanied by considerable toxicity . the emergence of immune "
    "checkpoint inhibitors targeting the programmed cell death protein 1 ( pd - 1 ) and its ligand "
    "( pd - l1 ) has transformed the therapeutic landscape of advanced nsclc . in this study , we "
    "evaluated the efficacy and safety of a pd - 1 inhibitor combined with platinum based "
    "chemotherapy as first line treatment in patients with metastatic nsclc without targetable driver "
    "mutations . a total of 616 patients were randomly assigned in a 2 : 1 ratio to receive either "
    "the combination of immunotherapy plus chemotherapy or chemotherapy alone . the primary endpoints "
    "were overall survival and progression free survival . patients were stratified according to "
    "pd - l1 expression level , histology , and smoking status . after a median follow up of 22 "
    "months , the combination group showed a significantly longer overall survival than the "
    "chemotherapy group ( median 22.0 versus 10.7 months ; hazard ratio for death 0.56 , 95% "
    "confidence interval 0.45 to 0.70 ; p < 0.001 ) . the benefit was observed across all pd - l1 "
    "subgroups , including patients with low or absent expression . progression free survival was "
    "also improved ( median 8.8 versus 4.9 months ) , and the overall response rate nearly doubled . "
    "immune related adverse events , including pneumonitis , colitis , and thyroid dysfunction , were "
    "more frequent in the combination group but were mostly manageable with corticosteroids and "
    "treatment interruption . these results indicate that adding pd - 1 blockade to standard "
    "chemotherapy provides a substantial survival benefit in previously untreated metastatic nsclc , "
    "and support its use as a first line standard of care ."
)

EXAMPLE_ALZHEIMER = (
    "alzheimer 's disease is the most common cause of dementia , affecting an estimated 50 million "
    "people worldwide , a number expected to triple by 2050 as populations age . the disease is "
    "characterized by the progressive accumulation of extracellular amyloid beta plaques and "
    "intracellular neurofibrillary tangles of hyperphosphorylated tau protein , leading to synaptic "
    "loss and neuronal death . despite decades of research , there is still no therapy that halts or "
    "reverses the underlying neurodegenerative process , and available treatments offer only modest "
    "symptomatic relief . early and accurate diagnosis remains a major challenge , as clinical "
    "symptoms often appear years after the onset of pathological changes . in this study , we "
    "investigated the diagnostic value of plasma biomarkers for the early detection of alzheimer 's "
    "disease in a cohort of 480 older adults , including cognitively normal individuals , patients "
    "with mild cognitive impairment , and patients with established dementia . blood samples were "
    "analyzed for phosphorylated tau 181 ( p - tau181 ) , the amyloid beta 42 / 40 ratio , and "
    "neurofilament light chain , and the results were compared with cerebrospinal fluid measurements "
    "and amyloid positron emission tomography , which served as reference standards . participants "
    "were followed for a mean of three years to assess cognitive decline . plasma p - tau181 showed a "
    "strong correlation with amyloid pet status and discriminated patients with alzheimer 's disease "
    "from cognitively normal controls with an area under the curve of 0.89 . elevated baseline "
    "p - tau181 levels predicted progression from mild cognitive impairment to dementia , "
    "independently of age , sex , and apolipoprotein e genotype . the amyloid beta 42 / 40 ratio "
    "provided additional discriminative value , while neurofilament light chain reflected the "
    "intensity of neurodegeneration but was less specific . these findings suggest that plasma "
    "biomarkers , and particularly p - tau181 , offer a minimally invasive , scalable , and cost "
    "effective tool for the early identification of alzheimer 's disease , with important "
    "implications for screening and for the enrollment of patients in disease modifying trials ."
)

EXAMPLE_AMR = (
    "antimicrobial resistance has become one of the most pressing threats to global public health , "
    "and is projected to cause up to 10 million deaths per year by 2050 if current trends continue . "
    "the inappropriate and excessive use of broad spectrum antibiotics , both in human medicine and "
    "in agriculture , has accelerated the emergence of multidrug resistant bacteria , particularly "
    "among gram negative organisms responsible for hospital acquired infections . carbapenem "
    "resistant enterobacteriaceae are of special concern , as they are associated with high mortality "
    "and very limited treatment options . in this study , we examined the impact of an antimicrobial "
    "stewardship program on resistance rates and clinical outcomes in a 600 bed tertiary hospital "
    "over a four year period . the program included prospective audit and feedback , restriction of "
    "selected broad spectrum agents , mandatory infectious disease consultation for certain "
    "prescriptions , and continuous education of prescribers . antibiotic consumption was measured in "
    "defined daily doses per 100 patient days , and resistance rates were monitored through routine "
    "microbiological surveillance . a total of 18,420 patients receiving antibiotics were included in "
    "the analysis . after implementation of the program , overall antibiotic consumption decreased by "
    "21% , with the most pronounced reduction observed for carbapenems and third generation "
    "cephalosporins . the incidence of carbapenem resistant enterobacteriaceae fell from 4.8 to 2.6 "
    "cases per 1000 admissions , and the rate of clostridioides difficile infection was also "
    "significantly reduced . importantly , these reductions were achieved without any increase in "
    "mortality , length of hospital stay , or readmission rates , indicating that more restrictive "
    "prescribing did not compromise patient safety . these results demonstrate that a comprehensive "
    "antimicrobial stewardship program can effectively curb the spread of multidrug resistant "
    "organisms while preserving clinical outcomes , and reinforce the need for such interventions as "
    "a core component of hospital infection control ."
)

EXAMPLES = [
    [EXAMPLE_DIABETES],
    [EXAMPLE_NSCLC],
    [EXAMPLE_ALZHEIMER],
    [EXAMPLE_AMR],
]

DESCRIPTION = """
# 🧬 Sumarizacao de Abstracts Biomedicos — BioBART + PubMed

Modelo **BioBART** ajustado (*fine-tuning*) no dataset **PubMed Summarization** para gerar um
**resumo conciso** a partir do corpo de um artigo cientifico biomedico.

Projeto final da disciplina *Topicos em Processamento de Linguagem Natural*.
Cole um texto em ingles (ou clique em um dos exemplos abaixo) e clique em **Sumarizar**.
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
    gr.Examples(
        examples=EXAMPLES,
        inputs=[inp],
        label="Exemplos de artigos longos (diabetes, cancer de pulmao, alzheimer, resistencia a antibioticos)",
    )

    gr.Markdown(
        f"Modelo: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) · "
        "Base: `GanjinZero/biobart-v2-base` · Dataset: `ccdv/pubmed-summarization`"
    )

if __name__ == "__main__":
    demo.launch()
