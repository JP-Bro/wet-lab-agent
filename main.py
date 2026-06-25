import re
from src.agent.graph import build_graph, tracker
from src.agent.state import AgentState

def run_agent(question: str, initial_hypothesis: str):
    print(f"\n{'='*50}")
    print(f"QUESTION: {question}")
    print(f"HYPOTHESIS: {initial_hypothesis}")
    print(f"{'='*50}")

    tracker.start_run(question, initial_hypothesis)

    initial_state: AgentState = {
        "question": question,
        "hypothesis": initial_hypothesis,
        "confidence": 0.1,
        "experiments": [],
        "iteration": 1,
        "literature_context": "",
        "next_experiment": {},
        "latest_result": "",
        "final_report": None,
        "finished": False,
        "reasoning_trace": []
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL REPORT:")
    print("="*50)

    # strip <think> tags
    clean_report = re.sub(
        r'<think>.*?</think>', '',
        final_state["final_report"],
        flags=re.DOTALL
    ).strip()
    print(clean_report)

    print("\n" + "="*50)
    print("REASONING TRACE:")
    print("="*50)
    for step in final_state["reasoning_trace"]:
        print(f"  • {step}")

    print(f"\nFinal confidence: {final_state['confidence']:.2f}")
    print(f"Total iterations: {final_state['iteration']}")

    return final_state

if __name__ == "__main__":
    run_agent(
        question="Which kinase drives EGFR inhibitor resistance in NSCLC cell lines?",
        initial_hypothesis="EGFR inhibitor resistance in NSCLC is driven by MET amplification activating downstream PI3K/AKT signaling"
    )