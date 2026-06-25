from langgraph.graph import StateGraph, END
from src.agent.state import AgentState, ExperimentResult
from src.tools.literature import search_literature
from src.tools.experiment import design_experiment
from src.tools.feasibility import check_feasibility
from src.tools.result_parser import parse_result
from src.tools.hypothesis import update_hypothesis
from src.agent.critic import critique_hypothesis
from src.memory.tracker import ExperimentTracker
from config.settings import MAX_ITERATIONS, CONFIDENCE_THRESHOLD
from groq import Groq
from config.settings import GROQ_API_KEY, MODEL_NAME
import json
import re

client = Groq(api_key=GROQ_API_KEY)
tracker = ExperimentTracker()

def literature_node(state: AgentState) -> AgentState:
    print(f"\n[Node 1] Searching literature...")
    context = search_literature(state["hypothesis"])
    state["literature_context"] = context
    state["reasoning_trace"].append(f"Literature searched: {context[:200]}...")
    print(f"  → Found context: {context[:100]}...")
    return state

def design_node(state: AgentState) -> AgentState:
    print(f"\n[Node 2] Designing experiment (iteration {state['iteration']})...")
    previous = ""
    if state["experiments"]:
        last = state["experiments"][-1]
        previous = last.findings
    experiment = design_experiment(
        hypothesis=state["hypothesis"],
        iteration=state["iteration"],
        previous_findings=previous
    )
    state["next_experiment"] = experiment
    state["reasoning_trace"].append(
        f"Designed: {experiment.get('assay_type')} on {experiment.get('cell_line')} — {experiment.get('rationale','')[:80]}"
    )
    print(f"  → Assay: {experiment.get('assay_type')} | Cell line: {experiment.get('cell_line')}")
    print(f"  → Rationale: {experiment.get('rationale','')[:80]}")
    return state

def feasibility_node(state: AgentState) -> AgentState:
    print(f"\n[Node 3] Checking feasibility...")
    result = check_feasibility(state["next_experiment"])
    if result["issues"]:
        print(f"  → Issues found: {result['issues']}")
        state["next_experiment"] = result["adjusted_experiment"]
        state["reasoning_trace"].append(f"Feasibility issues fixed: {result['issues']}")
    else:
        print(f"  → Experiment is feasible")
        state["reasoning_trace"].append("Feasibility check passed")
    return state

def result_node(state: AgentState) -> AgentState:
    print(f"\n[Node 4] Running experiment and parsing results...")
    parsed = parse_result(state["next_experiment"], state["iteration"])
    state["latest_result"] = parsed["key_finding"]
    state["reasoning_trace"].append(f"Finding: {parsed['key_finding']}")

    exp_record = ExperimentResult(
        iteration=state["iteration"],
        assay_type=state["next_experiment"].get("assay_type", "unknown"),
        conditions=state["next_experiment"],
        findings=parsed["key_finding"],
        confidence_delta=parsed["confidence_delta"]
    )
    state["experiments"].append(exp_record)
    print(f"  → Finding: {parsed['key_finding']}")
    if parsed.get("ic50_nm"):
        print(f"  → IC50: {parsed['ic50_nm']} nM | R²: {parsed.get('r_squared')}")
    return state

def hypothesis_node(state: AgentState) -> AgentState:
    print(f"\n[Node 5] Updating hypothesis...")

    last_exp = state["experiments"][-1]
    updated = update_hypothesis(
        current_hypothesis=state["hypothesis"],
        current_confidence=state["confidence"],
        latest_finding=state["latest_result"],
        confidence_delta=last_exp.confidence_delta,
        iteration=state["iteration"]
    )

    state["hypothesis"] = updated["refined_hypothesis"]
    new_confidence = updated["new_confidence"]

    # critic agent
    all_findings = [e.findings for e in state["experiments"]]
    critique = critique_hypothesis(
        hypothesis=state["hypothesis"],
        confidence=new_confidence,
        latest_finding=state["latest_result"],
        iteration=state["iteration"],
        all_findings=all_findings
    )

    # enforce penalty
    penalty = float(critique.get("confidence_penalty", 0.08))
    if critique.get("weaknesses") and penalty < 0.05:
        penalty = 0.08
    penalty = max(0.05, min(0.15, penalty))

    new_confidence = max(0.05, new_confidence - penalty)
    state["confidence"] = new_confidence
    state["iteration"] += 1

    verdict = critique.get("verdict", "unknown")
    weaknesses = critique.get("weaknesses", [])
    alt = critique.get("alternative_hypothesis", "")

    state["reasoning_trace"].append(
        f"Critic [{verdict}] penalty:-{penalty:.2f} | {weaknesses[0] if weaknesses else 'No issues'}"
    )
    state["reasoning_trace"].append(
        f"Confidence after critique: {new_confidence:.2f}"
    )

    # store critique for UI
    if "_critiques" not in state:
        state["_critiques"] = []

    tracker.log_iteration(
        iteration=state["iteration"],
        experiment=state["next_experiment"],
        finding=state["latest_result"],
        confidence=state["confidence"]
    )

    print(f"  → Confidence after critique: {new_confidence:.2f} (penalty: -{penalty:.2f})")
    print(f"  → Critic verdict: {verdict}")
    print(f"  → Weakness: {weaknesses[0] if weaknesses else 'none'}...")
    return state

def report_node(state: AgentState) -> AgentState:
    print(f"\n[Node 6] Generating final report...")

    experiments_summary = "\n".join([
        f"  Iteration {e.iteration}: {e.assay_type} on {e.conditions.get('cell_line','?')} → {e.findings}"
        for e in state["experiments"]
    ])

    prompt = f"""You are a scientist writing a final experimental report.

Original Question: {state['question']}
Final Hypothesis: {state['hypothesis']}
Final Confidence: {state['confidence']:.2f}
Total Iterations: {state['iteration']}

Experiments Conducted:
{experiments_summary}

Write a concise scientific report with these exact sections:
1. Summary
2. Methods (brief)
3. Key Findings
4. Conclusion
5. Recommended Next Steps

Be specific about IC50 values, cell lines, and compounds used.
Reflect the confidence level honestly — do not oversell weak evidence."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    raw_report = response.choices[0].message.content
    # strip think tags
    clean_report = re.sub(r'<think>.*?</think>', '', raw_report, flags=re.DOTALL).strip()

    state["final_report"] = clean_report
    state["finished"] = True

    tracker.log_final(clean_report, state["iteration"], state["confidence"])
    tracker.end_run()

    print(f"\n{'='*50}")
    print("FINAL REPORT GENERATED")
    print('='*50)
    return state

def should_continue(state: AgentState) -> str:
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        print(f"\n✅ Confidence threshold reached ({state['confidence']:.2f}). Generating report.")
        return "report"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n⚠️ Max iterations reached. Generating report.")
        return "report"
    print(f"\n🔄 Continuing loop (iteration {state['iteration']})...")
    return "continue"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("literature", literature_node)
    graph.add_node("design", design_node)
    graph.add_node("feasibility", feasibility_node)
    graph.add_node("result", result_node)
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("literature")
    graph.add_edge("literature", "design")
    graph.add_edge("design", "feasibility")
    graph.add_edge("feasibility", "result")
    graph.add_edge("result", "hypothesis")
    graph.add_edge("report", END)

    graph.add_conditional_edges(
        "hypothesis",
        should_continue,
        {
            "continue": "design",
            "report": "report"
        }
    )

    return graph.compile()