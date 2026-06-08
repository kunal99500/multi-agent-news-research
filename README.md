# DISPATCH — Autonomous News Intelligence

A multi-agent system that researches a topic end to end: it searches the web,
distills the sources, writes a news article, and then critiques its own draft —
looping until the story holds up. Built with LangChain 1.0, LangGraph, and a
Streamlit front end.

---

## What it does

You type a query. Four agents take over:

1. **Search agent** — queries the web (Tavily) and returns the top results.
2. **Reader agent** — extracts and summarizes the key information from those results.
3. **Writer chain** — drafts a structured news article grounded in the research.
4. **Critic chain** — scores the draft across six dimensions and decides whether it
   needs revision. If it does, its instructions are fed back to the writer and the
   draft is rewritten — up to a capped number of passes.

The result is a written article plus a transparent quality verdict: an overall
score, per-dimension breakdown, strengths, weaknesses, and any missing information.

---

## Architecture

```
            ┌──────────────┐
   query ──▶│ Search Agent │  web search (Tavily)
            └──────┬───────┘
                   │ raw results
            ┌──────▼───────┐
            │ Reader Agent │  extract + summarize
            └──────┬───────┘
                   │ distilled research
            ┌──────▼───────┐
            │ Writer Chain │◀──────────────┐
            └──────┬───────┘               │
                   │ draft article         │ revision
            ┌──────▼───────┐               │ instructions
            │ Critic Chain │───────────────┘
            └──────┬───────┘
                   │ approved (or max revisions reached)
                   ▼
          final article + scored verdict
```

The writer ⇄ critic loop is the core idea: the system grades its own output and
revises against the critic's findings, rather than returning the first draft.

---

## Tech stack

- **Orchestration:** LangChain 1.0, LangGraph (`create_agent`)
- **Models:** OpenAI `gpt-4o-mini` via `langchain-openai`
- **Structured output:** Pydantic schemas with `with_structured_output` — the critic
  returns a typed object, not parsed text
- **Search:** Tavily
- **Scraping:** Requests + BeautifulSoup
- **Front end:** Streamlit

---

## Project structure

```
multi_agent/
├── agents.py        # agents + writer/critic chains and prompts
├── pipeline.py      # orchestration: the full search→read→write→critique loop
├── tools.py         # web search + page scraping tools
├── app.py           # Streamlit dashboard
├── requirements.txt
├── .env.example     # template for required API keys
└── README.md
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API keys**

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
```

---

## Running it

**Command line:**

```bash
python pipeline.py
```

**Web app:**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter a query, watch each agent report
in live, and review the article alongside the critic's scored verdict.

---

## How the critic works

The critic returns a structured `CriticResponse` with:

- **Scores** (0.0–1.0): relevance, completeness, accuracy, source quality, clarity, overall
- **Strengths / weaknesses / missing information**
- **`hallucination_detected`** and **`needs_revision`** flags
- **`revision_instructions`** — fed back into the writer when revision is needed

Because the output is a typed Pydantic object, the rest of the pipeline reads it as
data (`critic.score.overall`, `critic.needs_revision`) with no string parsing.

---

## Roadmap

- Source-date tagging end to end so recency is verifiable, not asserted
- Claim-level citation in the writer, with the critic flagging unsupported claims
- A router agent to decide between *re-search* and *rewrite* on revision
- Swappable model tiers (e.g. routing some stages through cheaper/free models)

---

## Notes

- API keys live in `.env` (gitignored) locally, and in **Streamlit Secrets** when
  deployed — never in the repo.
- Scraping major news sites with plain HTTP is unreliable (paywalls, bot blocks);
  the pipeline leans on the search provider's extracted content for those cases.