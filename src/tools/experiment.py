from groq import Groq
from config.settings import GROQ_API_KEY, MODEL_NAME
import json

client = Groq(api_key=GROQ_API_KEY)

def design_experiment(hypothesis: str, iteration: int, previous_findings: str = "", literature_context: str = "") -> dict:
    """Design single-compound experiments that exist in GDSC2."""

    # SEQUENCE: cell line + single compound pairs
    sequence = [
        {
            "cell_line": "PC9-MET",
            "compound": "Crizotinib",
            "rationale": "Test if MET inhibition is effective in MET-amplified resistant cells"
        },
        {
            "cell_line": "PC9-MET",
            "compound": "Savolitinib",
            "rationale": "Alternative MET inhibitor — test if more potent than Crizotinib"
        },
        {
            "cell_line": "HCC827",
            "compound": "Gefitinib",
            "rationale": "EGFR-sensitive baseline — confirm normal EGFR inhibitor response"
        },
        {
            "cell_line": "NCI-H1975",
            "compound": "Osimertinib",
            "rationale": "T790M-resistant line — test if can overcome EGFR resistance"
        },
        {
            "cell_line": "A549",
            "compound": "Gefitinib",
            "rationale": "KRAS-mutant negative control — test if EGFR inhibition fails in EGFR-independent line"
        },
        {
            "cell_line": "H1993",
            "compound": "Capmatinib",
            "rationale": "Independent MET-amplified line — validate MET-inhibitor sensitivity"
        },
        {
            "cell_line": "PC9-MET",
            "compound": "Capmatinib",
            "rationale": "Best-in-class MET inhibitor — determine optimal therapy"
        },
    ]

    idx = min(iteration - 1, len(sequence) - 1)
    assigned = sequence[idx]

    assay_sequence = ["Cell Viability Assay", "Cell Viability Assay", "Western Blot", "Western Blot", "Flow Cytometry", "Western Blot", "Cell Viability Assay"]
    assay = assay_sequence[idx]

    prompt = f"""You are an expert wet lab scientist. Design a SINGLE compound experiment.

HYPOTHESIS: {hypothesis}
ITERATION: {iteration}
CELL LINE: {assigned['cell_line']}
COMPOUND: {assigned['compound']}
ASSAY: {assay}

RULES:
1. You MUST test {assigned['compound']} on {assigned['cell_line']} — no substitutions.
2. Use {assay} as the readout method.
3. Design standard dose-response curve with 4-5 concentrations.
4. Include proper controls: vehicle (DMSO) + positive control.
5. Readout depends on assay:
   - Cell Viability: IC50, viability %
   - Western Blot: p-MET, p-AKT, p-EGFR levels
   - Flow Cytometry: apoptosis %, cell death %
6. Make rationale 1 sentence connecting to hypothesis.

RESPOND ONLY WITH VALID JSON:
{{
    "assay_type": "{assay}",
    "cell_line": "{assigned['cell_line']}",
    "compound": "{assigned['compound']}",
    "concentrations": ["1nM", "10nM", "100nM", "1uM"],
    "controls": ["vehicle (DMSO)", "positive control"],
    "readout": "IC50 / protein levels / apoptosis %",
    "rationale": "{assigned['rationale']}"
}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a precise experimental biologist. Follow the rules exactly. Return only valid JSON."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
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
            "assay_type": assay,
            "cell_line": assigned["cell_line"],
            "compound": assigned["compound"],
            "concentrations": ["1nM", "10nM", "100nM", "1uM"],
            "controls": ["DMSO", "positive control"],
            "readout": "IC50" if "Cell Viability" in assay else "protein levels",
            "rationale": assigned["rationale"]
        }