prompt = f"""You are a scientist analyzing evidence for a hypothesis.

Hypothesis: {current_hypothesis}
Current confidence: {current_confidence:.2f}
New finding: {latest_finding}

Respond ONLY with valid JSON:
{{
    "refined_hypothesis": "same or refined hypothesis",
    "reasoning": "one sentence",
    "confidence_justified": true
}}

Confidence rules:
- If finding DIRECTLY supports hypothesis: delta +0.20
- If finding indirectly supports: delta +0.15
- If finding is mixed: delta +0.08
- If finding contradicts: delta -0.05"""