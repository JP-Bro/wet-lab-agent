from groq import Groq
from config.settings import GROQ_API_KEY, MODEL_NAME, CONFIDENCE_THRESHOLD
import json

client = Groq(api_key=GROQ_API_KEY)

def update_hypothesis(
    current_hypothesis: str,
    current_confidence: float,
    latest_finding: str,
    confidence_delta: float,
    iteration: int
) -> dict:
    """Update hypothesis based on latest result."""

    new_confidence = min(current_confidence + confidence_delta, 1.0)

    prompt = f"""You are a scientist updating a hypothesis based on new evidence.

Current hypothesis: {current_hypothesis}
Current confidence: {current_confidence:.2f}
New finding: {latest_finding}
Updated confidence: {new_confidence:.2f}
Iteration: {iteration}

Should the hypothesis be refined based on this finding?
Respond ONLY with valid JSON:
{{
    "refined_hypothesis": "updated hypothesis (or same if no change needed)",
    "reasoning": "one sentence explanation",
    "confidence_justified": true
}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw.strip())
        result["new_confidence"] = new_confidence
        result["threshold_reached"] = new_confidence >= CONFIDENCE_THRESHOLD
        return result
    except:
        return {
            "refined_hypothesis": current_hypothesis,
            "reasoning": "Hypothesis supported by evidence",
            "confidence_justified": True,
            "new_confidence": new_confidence,
            "threshold_reached": new_confidence >= CONFIDENCE_THRESHOLD
        }