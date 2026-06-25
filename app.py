import streamlit as st
import threading
import queue
import time
import plotly.graph_objects as go
from src.agent.graph import build_graph, tracker
from src.agent.state import AgentState

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WetLab AI Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: #0a0a0f; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b2e 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(88,166,255,0.05) 0%, transparent 60%);
    pointer-events: none;
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

/* Stat cards */
.stat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #58a6ff; }
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

/* Node status cards */
.node-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.node-running {
    border-left: 3px solid #f0883e;
    animation: pulse 1s infinite;
}
.node-done {
    border-left: 3px solid #3fb950;
}
.node-pending {
    border-left: 3px solid #30363d;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Finding cards */
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

/* Critic cards */
.critic-card {
    background: #161b22;
    border: 1px solid #f0883e44;
    border-left: 3px solid #f0883e;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.critic-verdict {
    font-size: 0.7rem;
    color: #f0883e;
    text-transform: uppercase;
    font-weight: 600;
}

/* Report section */
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

/* Input styling */
.stTextInput input, .stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 6px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}

/* Progress bar */
.confidence-bar-container {
    background: #21262d;
    border-radius: 20px;
    height: 8px;
    overflow: hidden;
    margin-top: 8px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff);
    transition: width 0.5s ease;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-running {
    background: rgba(240,136,62,0.15);
    color: #f0883e;
    border: 1px solid rgba(240,136,62,0.3);
}
.status-complete {
    background: rgba(63,185,80,0.15);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.3);
}
.status-idle {
    background: rgba(139,148,158,0.15);
    color: #8b949e;
    border: 1px solid rgba(139,148,158,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
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
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "iteration" not in st.session_state:
    st.session_state.iteration = 0
if "hypothesis" not in st.session_state:
    st.session_state.hypothesis = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
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
        help="The scientific question you want to investigate"
    )

    hypothesis = st.text_area(
        "Initial Hypothesis",
        value="EGFR inhibitor resistance in NSCLC is driven by MET amplification activating downstream PI3K/AKT signaling",
        height=100,
        help="Your starting hypothesis"
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        max_iter = st.number_input("Max Iterations", min_value=2, max_value=10, value=5)
    with col2:
        threshold = st.number_input("Confidence Threshold", min_value=0.5, max_value=0.99, value=0.75, step=0.05)

    st.markdown("<br/>", unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Agent", disabled=st.session_state.running)

    if st.session_state.running:
        st.markdown("""
        <div class='status-badge status-running'>⚡ Agent Running</div>
        """, unsafe_allow_html=True)
    elif st.session_state.finished:
        st.markdown("""
        <div class='status-badge status-complete'>✅ Complete</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='status-badge status-idle'>○ Idle</div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#21262d;margin:16px 0;'/>", unsafe_allow_html=True)
    st.markdown("**Stack**")
    stack_items = ["LangGraph", "Groq LLM", "ChromaDB RAG", "PubMed API", "MLflow", "scipy IC50"]
    for item in stack_items:
        st.markdown(f"<div style='color:#8b949e;font-size:0.8rem;padding:2px 0;'>▸ {item}</div>", unsafe_allow_html=True)

    if st.button("🗑️ Reset", disabled=st.session_state.running):
        for key in ["running","finished","confidence_history","findings","critiques",
                    "current_node","final_report","final_state","iteration","hypothesis"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🧬 Autonomous Wet Lab Agent</h1>
    <p>AI-powered closed-loop hypothesis validation · LangGraph · ChromaDB RAG · IC50 Curve Fitting · Multi-Agent Critic</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    conf = st.session_state.confidence_history[-1] if st.session_state.confidence_history else 0.0
    conf_percent = int(conf * 100)
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{conf_percent}%</div>
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

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Monitor", "🔬 Experiments", "🔍 Critic Analysis", "📄 Final Report"])

# ── Tab 1: Live Monitor ────────────────────────────────────────────────────────
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
                marker=dict(size=8, color='#58a6ff', line=dict(color='#0d1117', width=2)),
                fill='tozeroy',
                fillcolor='rgba(88,166,255,0.08)',
                name='Confidence'
            ))
            fig.add_hline(y=0.75, line_dash="dash", line_color="#3fb950",
                         annotation_text="Threshold", annotation_position="right")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b949e', family='Inter'),
                xaxis=dict(showgrid=True, gridcolor='#21262d', title='Iteration',
                          zeroline=False, tickfont=dict(color='#8b949e')),
                yaxis=dict(showgrid=True, gridcolor='#21262d', range=[0, 1],
                          title='Confidence', tickfont=dict(color='#8b949e')),
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.markdown("""
            <div style='background:#161b22;border:1px solid #21262d;border-radius:12px;
                        height:280px;display:flex;align-items:center;justify-content:center;
                        color:#8b949e;font-size:0.9rem;'>
                Confidence chart will appear here
            </div>""", unsafe_allow_html=True)

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
        for icon, label, node_id in nodes:
            if st.session_state.current_node == node_id:
                css = "node-running"
                dot = "🟠"
            elif st.session_state.finished:
                css = "node-done"
                dot = "🟢"
            else:
                css = "node-pending"
                dot = "⚪"
            st.markdown(f"""
            <div class='node-card {css}'>
                <span style='font-size:1.1rem;'>{icon}</span>
                <span style='color:#e6edf3;font-size:0.88rem;font-weight:500;'>{label}</span>
                <span style='margin-left:auto;'>{dot}</span>
            </div>""", unsafe_allow_html=True)

    # Current hypothesis
    if st.session_state.hypothesis:
        st.markdown("#### Current Hypothesis")
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #21262d;border-left:3px solid #58a6ff;
                    border-radius:8px;padding:16px;color:#e6edf3;font-size:0.9rem;line-height:1.6;'>
            {st.session_state.hypothesis}
        </div>""", unsafe_allow_html=True)

# ── Tab 2: Experiments ─────────────────────────────────────────────────────────
with tab2:
    if st.session_state.findings:
        st.markdown(f"#### {len(st.session_state.findings)} Experiments Conducted")
        for i, f in enumerate(st.session_state.findings):
            ic50_html = f"<div class='ic50-badge'>IC50 = {f['ic50']} nM</div>" if f.get('ic50') else ""
            st.markdown(f"""
            <div class='finding-card'>
                <div class='finding-iteration'>Iteration {f['iteration']} · {f['assay']}</div>
                <div class='finding-text'>{f['finding']}</div>
                {ic50_html}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#8b949e;'>
            <div style='font-size:2rem;margin-bottom:12px;'>🔬</div>
            <div>No experiments yet. Run the agent to start.</div>
        </div>""", unsafe_allow_html=True)

# ── Tab 3: Critic ──────────────────────────────────────────────────────────────
with tab3:
    if st.session_state.critiques:
        st.markdown("#### Critic Agent Analysis")
        for c in st.session_state.critiques:
            verdict_color = {"strong": "#3fb950", "moderate": "#f0883e", "weak": "#ff7b72"}.get(c.get('verdict','weak'), '#8b949e')
            penalty = c.get('penalty', 0.12)
            st.markdown(f"""
            <div class='critic-card'>
                <div class='critic-verdict'>
                    Iteration {c['iteration']} · <span style='color:{verdict_color};'>{c.get('verdict','?').upper()} EVIDENCE</span> · penalty: -{penalty:.2f}
                </div>
                <div style='color:#e6edf3;font-size:0.88rem;margin-top:8px;'>
                    <strong style='color:#8b949e;'>Weakness:</strong> {c.get('weakness','—')}
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#8b949e;'>
            <div style='font-size:2rem;margin-bottom:12px;'>🔍</div>
            <div>Critic analysis will appear after each iteration.</div>
        </div>""", unsafe_allow_html=True)
        
# ── Tab 4: Report ──────────────────────────────────────────────────────────────
with tab4:
    if st.session_state.final_report:
        st.markdown("#### Final Scientific Report")
        # strip <think> blocks from report
        import re
        clean_report = re.sub(r'<think>.*?</think>', '', st.session_state.final_report, flags=re.DOTALL).strip()
        st.markdown(f"""
        <div class='report-container'>{clean_report}</div>
        """, unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Report",
            data=clean_report,
            file_name="wetlab_agent_report.txt",
            mime="text/plain"
        )
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;color:#8b949e;'>
            <div style='font-size:2rem;margin-bottom:12px;'>📄</div>
            <div>Final report will appear when agent completes.</div>
        </div>""", unsafe_allow_html=True)

# ── Run agent ──────────────────────────────────────────────────────────────────
def run_agent_thread(question, hypothesis, result_queue):
    """Run agent in background thread, push updates to queue."""

    # patch graph nodes to push live updates
    import src.agent.graph as graph_module

    original_literature = graph_module.literature_node
    original_design = graph_module.design_node
    original_feasibility = graph_module.feasibility_node
    original_result = graph_module.result_node
    original_hypothesis = graph_module.hypothesis_node
    original_report = graph_module.report_node

    def patched_literature(state):
        result_queue.put({"type": "node", "node": "literature"})
        return original_literature(state)

    def patched_design(state):
        result_queue.put({"type": "node", "node": "design"})
        return original_design(state)

    def patched_feasibility(state):
        result_queue.put({"type": "node", "node": "feasibility"})
        return original_feasibility(state)

    def patched_result(state):
        result_queue.put({"type": "node", "node": "result"})
        s = original_result(state)
        if s["experiments"]:
            last = s["experiments"][-1]
            result_queue.put({
                "type": "finding",
                "iteration": last.iteration,
                "assay": last.assay_type,
                "finding": last.findings,
                "ic50": None
            })
        return s

    def patched_hypothesis(state):
        result_queue.put({"type": "node", "node": "hypothesis"})
        s = original_hypothesis(state)
        result_queue.put({
            "type": "critique",
            "iteration": s["iteration"] - 1,
            "iteration": s["iteration"],
            "hypothesis": s["hypothesis"]
        })
        # extract critique from reasoning trace
        for line in reversed(s["reasoning_trace"]):
            if "Critic [" in line:
                import re
                verdict_match = re.search(r'Critic \[(\w+)\]', line)
                verdict = verdict_match.group(1) if verdict_match else "unknown"
                weakness = line.split("]: ")[-1].split(" | Alt:")[0] if "]: " in line else line
                alt = line.split("Alt: ")[-1] if "Alt: " in line else ""
                result_queue.put({
                    "type": "critique",
                    "iteration": s["iteration"],
                    "verdict": verdict,
                    "weakness": weakness,
                    "alternative": alt,
                    "penalty": 0.0
                })
                break
        return s

    def patched_report(state):
        result_queue.put({"type": "node", "node": "report"})
        s = original_report(state)
        result_queue.put({"type": "done", "report": s["final_report"]})
        return s

    graph_module.literature_node = patched_literature
    graph_module.design_node = patched_design
    graph_module.feasibility_node = patched_feasibility
    graph_module.result_node = patched_result
    graph_module.hypothesis_node = patched_hypothesis
    graph_module.report_node = patched_report

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
        graph.invoke(initial_state)
    except Exception as e:
        result_queue.put({"type": "error", "message": str(e)})

    # restore
    graph_module.literature_node = original_literature
    graph_module.design_node = original_design
    graph_module.feasibility_node = original_feasibility
    graph_module.result_node = original_result
    graph_module.hypothesis_node = original_hypothesis
    graph_module.report_node = original_report

if run_btn and not st.session_state.running:
    # reset state
    st.session_state.running = True
    st.session_state.finished = False
    st.session_state.confidence_history = [0.1]
    st.session_state.findings = []
    st.session_state.critiques = []
    st.session_state.final_report = None
    st.session_state.current_node = "literature"
    st.session_state.iteration = 0
    st.session_state.hypothesis = hypothesis

    result_queue = queue.Queue()
    thread = threading.Thread(
        target=run_agent_thread,
        args=(question, hypothesis, result_queue),
        daemon=True
    )
    thread.start()
    st.session_state._thread = thread
    st.session_state._queue = result_queue
    st.rerun()

# ── Poll queue while running ───────────────────────────────────────────────────
if st.session_state.running:
    q = getattr(st.session_state, '_queue', None)
    if q:
        updated = False
        try:
            while True:
                msg = q.get_nowait()
                updated = True
                if msg["type"] == "node":
                    st.session_state.current_node = msg["node"]
                elif msg["type"] == "finding":
                    st.session_state.findings.append(msg)
                    st.session_state.iteration = msg["iteration"]
                elif msg["type"] == "confidence":
                    st.session_state.confidence_history.append(msg["value"])
                    st.session_state.hypothesis = msg["hypothesis"]
                elif msg["type"] == "critique":
                    st.session_state.critiques.append(msg)
                elif msg["type"] == "done":
                    st.session_state.final_report = msg["report"]
                    st.session_state.running = False
                    st.session_state.finished = True
                    st.session_state.current_node = None
                elif msg["type"] == "error":
                    st.error(f"Agent error: {msg['message']}")
                    st.session_state.running = False
        except queue.Empty:
            pass

        time.sleep(0.5)
        st.rerun()