import streamlit as st
import threading
import queue
import time
import plotly.graph_objects as go
from src.agent.graph import build_graph, tracker
from src.agent.state import AgentState

st.set_page_config(
    page_title="WetLab AI Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }
.main { background: #0a0a0f; }
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b2e 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 24px;
    text-align: center;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero p {
    color: #8b949e;
    font-size: 1rem;
    margin: 0;
}
.stat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.stat-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
.finding-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.finding-iteration {
    font-size: 0.7rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.finding-text {
    color: #e6edf3;
    font-size: 0.9rem;
    margin-top: 4px;
    line-height: 1.5;
}
.ic50-badge {
    display: inline-block;
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.3);
    color: #3fb950;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}
.critic-card {
    background: #161b22;
    border: 1px solid #f0883e44;
    border-left: 3px solid #f0883e;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.critic-verdict {
    font-size: 0.75rem;
    color: #f0883e;
    text-transform: uppercase;
    font-weight: 600;
}
.report-container {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 28px;
    color: #e6edf3;
    font-size: 0.9rem;
    line-height: 1.8;
    white-space: pre-wrap;
    max-height: 600px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "finished" not in st.session_state:
    st.session_state.finished = False
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []
if "findings" not in st.session_state:
    st.session_state.findings = []
if "critiques" not in st.session_state:
    st.session_state.critiques = []
if "current_node" not in st.session_state:
    st.session_state.current_node = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "hypothesis" not in st.session_state:
    st.session_state.hypothesis = ""

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:1.3rem;font-weight:700;color:#e6edf3;'>🧬 WetLab Agent</div>
        <div style='font-size:0.75rem;color:#8b949e;margin-top:2px;'>Autonomous Experiment Designer</div>
    </div>
    <hr style='border-color:#21262d;margin:12px 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("**Configure Run**")

    question = st.text_area(
        "Biological Question",
        value="Which kinase drives EGFR inhibitor resistance in NSCLC cell lines?",
        height=80,
    )

    hypothesis = st.text_area(
        "Initial Hypothesis",
        value="EGFR inhibitor resistance in NSCLC is driven by MET amplification activating downstream PI3K/AKT signaling",
        height=100,
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        max_iter = st.number_input("Max Iterations", min_value=2, max_value=10, value=7)
    with col2:
        threshold = st.number_input("Confidence Threshold", min_value=0.5, max_value=0.99, value=0.65, step=0.05)

    st.markdown("<br/>", unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Agent", disabled=st.session_state.running)

    if st.session_state.running:
        st.markdown('<div style="background:rgba(240,136,62,0.15);color:#f0883e;border:1px solid rgba(240,136,62,0.3);border-radius:20px;padding:4px 12px;font-size:0.75rem;font-weight:600;">⚡ Agent Running</div>', unsafe_allow_html=True)
    elif st.session_state.finished:
        st.markdown('<div style="background:rgba(63,185,80,0.15);color:#3fb950;border:1px solid rgba(63,185,80,0.3);border-radius:20px;padding:4px 12px;font-size:0.75rem;font-weight:600;">✅ Complete</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:rgba(139,148,158,0.15);color:#8b949e;border:1px solid rgba(139,148,158,0.3);border-radius:20px;padding:4px 12px;font-size:0.75rem;font-weight:600;">○ Idle</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d;margin:16px 0;'/>", unsafe_allow_html=True)
    st.markdown("**Stack**")
    for item in ["LangGraph", "Groq LLM", "ChromaDB RAG", "PubMed API", "MLflow", "scipy IC50"]:
        st.markdown(f"<div style='color:#8b949e;font-size:0.8rem;padding:2px 0;'>▸ {item}</div>", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🧬 Autonomous Wet Lab Agent</h1>
    <p>AI-powered closed-loop hypothesis validation · LangGraph · ChromaDB RAG · IC50 Curve Fitting · Multi-Agent Critic</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    conf = st.session_state.confidence_history[-1] if st.session_state.confidence_history else 0.0
    conf_pct = int(conf * 100)
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{conf_pct}%</div>
        <div class='stat-label'>Confidence</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{st.session_state.iteration}</div>
        <div class='stat-label'>Iterations</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{len(st.session_state.findings)}</div>
        <div class='stat-label'>Experiments</div>
    </div>""", unsafe_allow_html=True)

with c4:
    status = "Running" if st.session_state.running else ("Complete" if st.session_state.finished else "Idle")
    color = "#f0883e" if st.session_state.running else ("#3fb950" if st.session_state.finished else "#8b949e")
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number' style='color:{color};font-size:1.3rem;'>{status}</div>
        <div class='stat-label'>Status</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Monitor", "🔬 Experiments", "🔍 Critic Analysis", "📄 Final Report"])

with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### Confidence Trajectory")
        if st.session_state.confidence_history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(st.session_state.confidence_history))),
                y=st.session_state.confidence_history,
                mode='lines+markers',
                line=dict(color='#58a6ff', width=2.5),
                marker=dict(size=8, color='#58a6ff'),
                fill='tozeroy',
                fillcolor='rgba(88,166,255,0.08)',
            ))
            fig.add_hline(y=0.65, line_dash="dash", line_color="#3fb950", annotation_text="Threshold")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b949e'),
                xaxis=dict(title='Iteration', showgrid=True, gridcolor='#21262d'),
                yaxis=dict(title='Confidence', range=[0, 1], showgrid=True, gridcolor='#21262d'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.markdown("Confidence chart will appear here")

    with col_right:
        st.markdown("#### Agent Pipeline")
        nodes = [
            ("📚", "Literature Search", "literature"),
            ("🧪", "Experiment Design", "design"),
            ("✅", "Feasibility Check", "feasibility"),
            ("📈", "Result Parser", "result"),
            ("🧠", "Hypothesis Update", "hypothesis"),
            ("🔍", "Critic Agent", "critic"),
            ("📄", "Report Generator", "report"),
        ]
        for icon, label, _ in nodes:
            if st.session_state.finished:
                dot = "🟢"
            else:
                dot = "⚪"
            st.markdown(f"<div style='display:flex;gap:12px;align-items:center;padding:8px;background:#161b22;border-radius:8px;margin:4px 0;'><span>{icon}</span><span style='color:#e6edf3;font-size:0.9rem;'>{label}</span><span style='margin-left:auto;'>{dot}</span></div>", unsafe_allow_html=True)

    if st.session_state.hypothesis:
        st.markdown("#### Current Hypothesis")
        st.markdown(f"<div style='background:#161b22;border:1px solid #21262d;border-left:3px solid #58a6ff;border-radius:8px;padding:16px;color:#e6edf3;font-size:0.9rem;line-height:1.6;'>{st.session_state.hypothesis}</div>", unsafe_allow_html=True)

with tab2:
    if st.session_state.findings:
        st.markdown(f"#### {len(st.session_state.findings)} Experiments Conducted")
        for f in st.session_state.findings:
            ic50_html = f"<div class='ic50-badge'>IC50 = {f.get('ic50', 'N/A')} nM</div>" if f.get('ic50') else ""
            st.markdown(f"""
            <div class='finding-card'>
                <div class='finding-iteration'>Iteration {f['iteration']} · {f['assay']}</div>
                <div class='finding-text'>{f['finding']}</div>
                {ic50_html}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:60px;color:#8b949e;'>No experiments yet. Run the agent to start.</div>", unsafe_allow_html=True)

with tab3:
    if st.session_state.critiques:
        st.markdown("#### Critic Agent Analysis")
        for c in st.session_state.critiques:
            verdict_color = {"strong": "#3fb950", "moderate": "#f0883e", "weak": "#ff7b72"}.get(c.get('verdict','weak'), '#8b949e')
            penalty = c.get('penalty', 0.12)
            st.markdown(f"""
            <div class='critic-card'>
                <div class='critic-verdict'>Iteration {c['iteration']} · <span style='color:{verdict_color};'>{c.get('verdict','?').upper()} EVIDENCE</span> · penalty: -{penalty:.2f}</div>
                <div style='color:#e6edf3;font-size:0.85rem;margin-top:6px;'>{c.get('weakness','—')}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:60px;color:#8b949e;'>Critic analysis will appear after each iteration.</div>", unsafe_allow_html=True)

with tab4:
    if st.session_state.final_report:
        st.markdown("#### Final Scientific Report")
        st.markdown(f"<div class='report-container'>{st.session_state.final_report}</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Report", data=st.session_state.final_report, file_name="wetlab_report.txt", mime="text/plain")
    else:
        st.markdown("<div style='text-align:center;padding:60px;color:#8b949e;'>Final report will appear when agent completes.</div>", unsafe_allow_html=True)

# ── Run Agent ────────────────────────────────────────────────────
def run_agent_thread(question, hypothesis, result_queue):
    try:
        initial_state: AgentState = {
            "question": question,
            "hypothesis": hypothesis,
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
        tracker.start_run(question, hypothesis)
        graph = build_graph()
        final_state = graph.invoke(initial_state)
        
        result_queue.put({"type": "done", "state": final_state})
    except Exception as e:
        result_queue.put({"type": "error", "message": str(e)})

if run_btn and not st.session_state.running:
    st.session_state.running = True
    st.session_state.finished = False
    st.session_state.confidence_history = [0.1]
    st.session_state.findings = []
    st.session_state.critiques = []
    st.session_state.final_report = None
    st.session_state.iteration = 0
    st.session_state.hypothesis = hypothesis

    result_queue = queue.Queue()
    thread = threading.Thread(target=run_agent_thread, args=(question, hypothesis, result_queue), daemon=True)
    thread.start()
    st.session_state._queue = result_queue
    st.rerun()

if st.session_state.running:
    q = getattr(st.session_state, '_queue', None)
    if q:
        try:
            msg = q.get_nowait()
            if msg["type"] == "done":
                final = msg["state"]
                st.session_state.final_report = final["final_report"]
                st.session_state.confidence_history = [0.1] + [exp.confidence_delta + sum(e.confidence_delta for e in final["experiments"][:i]) for i, exp in enumerate(final["experiments"])]
                st.session_state.finished = True
                st.session_state.running = False
                st.session_state.iteration = final["iteration"]
                st.session_state.hypothesis = final["hypothesis"]
                st.rerun()
        except queue.Empty:
            time.sleep(0.5)
            st.rerun()