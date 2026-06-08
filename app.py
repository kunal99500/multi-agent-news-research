"""
DISPATCH — Autonomous News Intelligence
A Streamlit front-end for the multi-agent research pipeline.

Run with:
    streamlit run app.py

Requires (add to requirements.txt):
    streamlit>=1.36.0
"""

import streamlit as st

# Pipeline building blocks. We drive the stages here (instead of calling
# # run_research_pipline) so the UI can show live per-agent progress.
# from agents import build_search_agent, reader_agent, writer_chain, critic_chain

MAX_REVISIONS = 2


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DISPATCH · News Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Styling — newsroom intelligence terminal
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --bg:#0d0d0f; --bg2:#141417; --card:#161619;
  --ink:#f2ede3; --muted:#8c877d; --faint:#5a564f;
  --accent:#ff8a3d; --accent2:#e4572e;
  --line:rgba(242,237,227,.08);
  --green:#5ad17f; --amber:#ffb84d; --orange:#fb923c; --red:#f2645a;
}

/* strip default streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none!important;}
[data-testid="stHeader"]{background:transparent;height:0;}

.stApp{
  background:
    radial-gradient(900px 500px at 12% -5%, rgba(255,138,61,.10), transparent 60%),
    radial-gradient(800px 500px at 95% 8%, rgba(228,87,46,.08), transparent 55%),
    var(--bg);
  color:var(--ink);
  font-family:'Hanken Grotesk', sans-serif;
}
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.035;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.block-container{padding-top:2.2rem; padding-bottom:5rem; max-width:1180px; position:relative; z-index:1;}

/* ---- hero ---- */
.kicker{
  font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.42em;
  text-transform:uppercase; color:var(--accent); margin-bottom:.4rem;
}
.kicker .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);
  margin-right:.7em;box-shadow:0 0 10px var(--accent);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.title{
  font-family:'Fraunces', serif; font-weight:300; font-size:5.4rem; line-height:.92;
  letter-spacing:-.02em; margin:0; color:var(--ink);
}
.title em{font-style:italic; color:var(--accent); font-weight:500;}
.subtitle{color:var(--muted); font-size:1.05rem; max-width:46ch; margin:.9rem 0 0; line-height:1.5;}
.rule{height:1px;background:linear-gradient(90deg,var(--line),transparent);margin:2rem 0 1.4rem;}

/* ---- input ---- */
.stTextInput > div > div{background:var(--bg2);border:1px solid var(--line);border-radius:0;}
.stTextInput input{
  background:transparent!important;color:var(--ink)!important;
  font-family:'Hanken Grotesk',sans-serif;font-size:1.05rem;padding:1rem 1.1rem!important;
}
.stTextInput input::placeholder{color:var(--faint);}
.stTextInput > div > div:focus-within{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent);}

.stButton button{
  background:var(--accent);color:#1a1206;border:none;border-radius:0;
  font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:.15em;
  text-transform:uppercase;font-size:.8rem;padding:1rem 1.6rem;width:100%;
  transition:transform .15s ease, box-shadow .15s ease;
}
.stButton button:hover{transform:translateY(-2px);box-shadow:0 10px 30px -10px var(--accent);color:#1a1206;}
.stButton button:active{transform:translateY(0);}

/* ---- section labels ---- */
.sec{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.32em;
  text-transform:uppercase;color:var(--faint);margin:2.4rem 0 1rem;display:flex;align-items:center;gap:.8rem;}
.sec::after{content:"";flex:1;height:1px;background:var(--line);}

/* ---- score gauge ---- */
.verdict{display:grid;grid-template-columns:auto 1fr;gap:2.4rem;align-items:center;
  background:var(--card);border:1px solid var(--line);padding:2rem 2.2rem;}
.ring{width:148px;height:148px;border-radius:50%;display:grid;place-items:center;position:relative;}
.ring::before{content:"";position:absolute;inset:11px;border-radius:50%;background:var(--card);}
.ring .val{position:relative;font-family:'Fraunces',serif;font-size:2.7rem;font-weight:500;line-height:1;}
.ring .lab{position:relative;font-family:'JetBrains Mono',monospace;font-size:.6rem;
  letter-spacing:.25em;color:var(--muted);text-transform:uppercase;margin-top:.25rem;}
.bars{display:flex;flex-direction:column;gap:.85rem;}
.bar-row{display:grid;grid-template-columns:130px 1fr 48px;align-items:center;gap:1rem;}
.bar-row .name{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);}
.bar-track{height:7px;background:rgba(242,237,227,.06);border-radius:99px;overflow:hidden;}
.bar-fill{height:100%;border-radius:99px;animation:grow 1s cubic-bezier(.2,.8,.2,1) both;}
@keyframes grow{from{width:0!important;}}
.bar-row .num{font-family:'JetBrains Mono',monospace;font-size:.82rem;text-align:right;color:var(--ink);}

/* ---- chips ---- */
.chips{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:1.2rem;}
.chip{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  padding:.5rem .9rem;border:1px solid var(--line);display:flex;align-items:center;gap:.5rem;color:var(--muted);}
.chip b{color:var(--ink);font-weight:700;}
.chip . d{width:7px;height:7px;border-radius:50%;}

/* ---- finding cards ---- */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
.fcard{background:var(--card);border:1px solid var(--line);padding:1.3rem 1.4rem;border-top:2px solid var(--edge);}
.fcard h4{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;
  margin:0 0 .9rem;color:var(--edge);}
.fcard ul{margin:0;padding-left:1.1rem;}
.fcard li{font-size:.92rem;line-height:1.55;color:var(--ink);margin-bottom:.6rem;opacity:.9;}
.fcard .empty{color:var(--faint);font-size:.9rem;font-style:italic;}

/* ---- article ---- */
.article-wrap [data-testid="stMarkdownContainer"]{font-family:'Newsreader','Hanken Grotesk',serif;}
.article-wrap h1,.article-wrap h2,.article-wrap h3{font-family:'Fraunces',serif!important;font-weight:500;}
[data-testid="stExpander"]{border:1px solid var(--line)!important;background:var(--bg2);border-radius:0!important;}

/* revision pill */
.revnote{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.12em;color:var(--accent);
  text-transform:uppercase;margin-bottom:1rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_color(v: float) -> str:
    if v >= 0.85: return "var(--green)"
    if v >= 0.70: return "var(--amber)"
    if v >= 0.50: return "var(--orange)"
    return "var(--red)"


def ring_html(overall: float) -> str:
    pct = int(round(overall * 100))
    col = score_color(overall)
    deg = overall * 360
    return f"""
    <div class="ring" style="background:conic-gradient({col} {deg}deg, rgba(242,237,227,.06) {deg}deg);">
      <div class="val" style="color:{col};">{overall:.2f}</div>
      <div class="lab">Overall</div>
    </div>"""


def bars_html(score) -> str:
    dims = [
        ("Relevance", score.relevance),
        ("Completeness", score.completeness),
        ("Accuracy", score.accuracy),
        ("Source Qual.", score.source_quality),
        ("Clarity", score.clarity),
    ]
    rows = ""
    for name, v in dims:
        col = score_color(v)
        rows += f"""
        <div class="bar-row">
          <div class="name">{name}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{v*100:.0f}%;background:{col};"></div></div>
          <div class="num">{v:.2f}</div>
        </div>"""
    return f'<div class="bars">{rows}</div>'


def chips_html(critic) -> str:
    rev = "var(--amber)" if critic.needs_revision else "var(--green)"
    hal = "var(--red)" if critic.hallucination_detected else "var(--green)"
    rev_t = "Yes" if critic.needs_revision else "No"
    hal_t = "Detected" if critic.hallucination_detected else "None"
    return f"""
    <div class="chips">
      <div class="chip"><span class="d" style="background:{rev}"></span>Needs Revision&nbsp;<b>{rev_t}</b></div>
      <div class="chip"><span class="d" style="background:{hal}"></span>Hallucination&nbsp;<b>{hal_t}</b></div>
      <div class="chip"><span class="d" style="background:var(--accent)"></span>Revision Passes&nbsp;<b>{st.session_state.get('revisions',0)}</b></div>
    </div>"""


def finding_card(title: str, items: list, edge: str) -> str:
    if items:
        body = "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    else:
        body = '<div class="empty">None noted.</div>'
    return f'<div class="fcard" style="--edge:{edge};"><h4>{title}</h4>{body}</div>'


# ---------------------------------------------------------------------------
# Orchestration (mirrors pipeline.py, with live progress)
# ---------------------------------------------------------------------------
from pipeline import run_research_pipline   # add near your other imports

def run_with_progress(query: str) -> dict:
    with st.status("Deploying agents…", expanded=True) as status:
        result = run_research_pipline(query, on_step=status.write)
        status.update(label="Research complete", state="complete", expanded=False)
    return result

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="kicker"><span class="dot"></span>Multi-Agent Research Engine · Live</div>
    <h1 class="title">The newsroom<br>that <em>investigates</em> itself.</h1>
    <p class="subtitle">A search agent gathers, a reader distills, a writer drafts,
    and a critic grades — looping until the story holds up.</p>
    <div class="rule"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
c1, c2 = st.columns([4, 1])
with c1:
    query = st.text_input(
        "query", placeholder="What story should the agents investigate?",
        label_visibility="collapsed",
    )
with c2:
    go = st.button("Investigate")

if go and query.strip():
    try:
        st.session_state.result = run_with_progress(query.strip())
        st.session_state.revisions = st.session_state.result.get("revisions", 0)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
elif go:
    st.warning("Enter a query first.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res:
    critic = res["critic_response"]

    # --- verdict / scores ---
    st.markdown('<div class="sec">Critic Verdict</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict">{ring_html(critic.score.overall)}{bars_html(critic.score)}</div>'
        + chips_html(critic),
        unsafe_allow_html=True,
    )

    # --- findings ---
    st.markdown('<div class="sec">Findings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cards">'
        + finding_card("Strengths", critic.strengths, "var(--green)")
        + finding_card("Weaknesses", critic.weaknesses, "var(--amber)")
        + finding_card("Missing", critic.missing_information, "var(--red)")
        + "</div>",
        unsafe_allow_html=True,
    )
    if critic.revision_instructions:
        st.markdown('<div class="sec">Revision Directives</div>', unsafe_allow_html=True)
        st.markdown(
            finding_card("Instructions", critic.revision_instructions, "var(--accent)"),
            unsafe_allow_html=True,
        )

    # --- the article ---
    st.markdown('<div class="sec">The Dispatch</div>', unsafe_allow_html=True)
    if res.get("revisions"):
        st.markdown(
            f'<div class="revnote">Refined across {res["revisions"]} critic-driven revision pass(es)</div>',
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        st.markdown('<div class="article-wrap">', unsafe_allow_html=True)
        st.markdown(res["report"])
        st.markdown("</div>", unsafe_allow_html=True)

    # --- raw agent trace ---
    st.markdown('<div class="sec">Agent Trace</div>', unsafe_allow_html=True)
    with st.expander("Search agent — raw results"):
        st.markdown(res["search_agent_response"])
    with st.expander("Reader agent — extracted summary"):
        st.markdown(res["scrapped_info"])
else:
    st.markdown(
        '<p style="color:var(--faint);font-family:\'JetBrains Mono\',monospace;'
        'font-size:.8rem;letter-spacing:.1em;margin-top:1.5rem;">'
        'AWAITING ASSIGNMENT — enter a query to dispatch the agents.</p>',
        unsafe_allow_html=True,
    )