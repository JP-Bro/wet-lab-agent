from typing import TypedDict, List, Optional
from pydantic import BaseModel

class ExperimentResult(BaseModel):
    iteration: int
    assay_type: str
    conditions: dict
    findings: str
    confidence_delta: float

class AgentState(TypedDict):
    # The original scientific question
    question: str
    
    # Current hypothesis being tested
    hypothesis: str
    
    # Confidence in hypothesis (0.0 to 1.0)
    confidence: float
    
    # All experiments run so far
    experiments: List[ExperimentResult]
    
    # Current iteration number
    iteration: int
    
    # Literature context retrieved
    literature_context: str
    
    # Next experiment designed
    next_experiment: dict
    
    # Latest result parsed
    latest_result: str
    
    # Final report
    final_report: Optional[str]
    
    # Is the agent done?
    finished: bool
    
    # Reasoning trace (full audit log)
    reasoning_trace: List[str]