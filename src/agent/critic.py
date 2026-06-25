from groq import Groq
from config.settings import GROQ_API_KEY, MODEL_NAME
import json
import time

client = Groq(api_key=GROQ_API_KEY)

def critique_hypothesis(
    hypothesis: str,
    confidence: float,
    latest_finding: str,
    iteration: int,
    all_findings: list[str]
) -> dict:

    findings_summary = "\n".join([f"  - Iter {i+1}: {f}" for i, f in enumerate(all_findings)])

    prompt = f"""You are a rigorous scientific critic reviewing experimental evidence.

Current hypothesis: {hypothesis}
Current confidence: {confidence:.2f}
Iteration: {iteration}

All findings so far:
{findings_summary}

Latest finding: {latest_finding}

Your job is to CHALLENGE this hypothesis. Look for:
1. Alternative explanations for the data
2. Confounding factors (wrong cell lines, missing controls)
3. Missing controls or replication
4. Weaknesses in experimental design
5. Other kinases or mechanisms that could explain the data

Respond ONLY with valid JSON:
{{
    "weaknesses": ["specific weakness 1", "specific weakness 2"],
    "alternative_hypothesis": "one specific alternative explanation",
    "confidence_penalty": 0.08,
    "critical_experiment_needed": "one specific experiment that would definitively prove/disprove",
    "verdict": "weak|moderate|strong"
}}

RULES:
- confidence_penalty MUST be between 0.05 and 0.15. NEVER return 0.0.
- verdict = weak if single cell line or no controls
- verdict = moderate if pathway confirmed but mechanism unclear
- verdict = strong only if MET dependency genetically validated with knockdown/knockout"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a devil's advocate scientist. Always find flaws. Always return confidence_penalty between 0.05 and 0.15. Never return 0.0."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw.strip())

            # enforce non-zero penalty
            if result.get("weaknesses") and result.get("confidence_penalty", 0) < 0.05:
                result["confidence_penalty"] = 0.08

            # clamp to valid range
            result["confidence_penalty"] = max(0.05, min(0.15, float(result.get("confidence_penalty", 0.08))))

            return result

        except Exception as e:
            if attempt == 2:
                return {
                    "weaknesses": [f"Critique parse failed: {str(e)}"],
                    "alternative_hypothesis": "Insufficient data to determine alternative",
                    "confidence_penalty": 0.08,
                    "critical_experiment_needed": "Repeat with proper positive and negative controls",
                    "verdict": "weak"
                }
            time.sleep(2 ** attempt)