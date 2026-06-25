from groq import Groq
from config.settings import GROQ_API_KEY, MODEL_NAME
import json

client = Groq(api_key=GROQ_API_KEY)

def design_experiment(hypothesis: str, iteration: int, previous_findings: str = "") -> dict:
    """Design the next experiment grounded in the actual hypothesis."""

    context = f"Previous findings: {previous_findings}" if previous_findings else "This is the first experiment."

    prompt = f"""You are an expert wet lab scientist. Design a focused experiment.

HYPOTHESIS TO TEST: {hypothesis}
ITERATION: {iteration}
{context}

CRITICAL RULES:
1. Read the hypothesis carefully. Design an experiment that directly tests it.
2. If the hypothesis is about NSCLC/lung cancer: ONLY use H1975, PC9, HCC827, A549, PC9-MET.
3. If the hypothesis is about a specific pathway (e.g. PI3K/AKT): measure that pathway directly.
4. If the hypothesis is about a kinase: test inhibitors of THAT kinase, not random compounds.
5. NEVER use HeLa unless the hypothesis is explicitly about cervical cancer.
6. Each iteration should build on previous findings — don't repeat the same experiment.
7. Vary assay types: use Western Blot for mechanism, Cell Viability for functional effect, Co-IP for interactions.

Respond ONLY with valid JSON:
{{
    "assay_type": "Western Blot | Cell Viability Assay | ELISA | Co-IP | Flow Cytometry",
    "cell_line": "specific cell line matching the hypothesis",
    "compound": "specific compound targeting the hypothesized kinase",
    "concentrations": ["1nM", "10nM", "100nM", "1uM", "10uM"],
    "controls": ["vehicle control (DMSO)", "positive control"],
    "readout": "specific protein/pathway being measured",
    "rationale": "one sentence linking this experiment to the hypothesis"
}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a domain expert scientist. Always design experiments that directly test the stated hypothesis. Never default to generic assays."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw.strip())
    except:
        return {
            "assay_type": "Western Blot",
            "cell_line": "PC9-MET",
            "compound": "Crizotinib",
            "concentrations": ["10nM", "100nM", "1uM", "10uM"],
            "controls": ["DMSO", "positive control"],
            "readout": "p-MET, p-AKT (Ser473)",
            "rationale": "Direct test of MET inhibition on downstream AKT signaling"
        }