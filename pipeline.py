from agents import build_search_agent, reader_agent, writer_chain, critic_chain
from datetime import date
MAX_REVISIONS = 2  # cap so a stubborn critic can't loop forever


def run_research_pipline(query: str, on_step=None) -> dict:
    """
    Run the multi-agent research pipeline.

    on_step: optional progress callback, on_step(message: str).
             The CLI passes `print`; the Streamlit app passes `st.status.write`.
             Defaults to a no-op.
    """
    step = on_step or (lambda msg: None)
    state = {}

    # ---- Search agent ----
    step("◆ Search agent — scanning the web…")
    search_agent = build_search_agent()
    search_agent_response = search_agent.invoke({
        "messages": [("user", f"Search the web for the following query and return the top 5 results: {query}")]
    })
    state["search_agent_response"] = search_agent_response["messages"][-1].content

    # ---- Reader agent ----
    step("◆ Reader agent — extracting & summarizing…")
    reader_agents = reader_agent()
    reader_agent_response = reader_agents.invoke({
        "messages": [("user",
                      f"Extract and summarize the key information of the query: {query}, "
                      f"from the following search results: {state['search_agent_response'][:6000]}")]
    })
    state["scrapped_info"] = reader_agent_response["messages"][-1].content

    research_combined = "\n".join([
        f"Search Results: {state['search_agent_response']}",
        f"Scrapped Information: {state['scrapped_info']}",
    ])

    # ---- Writer + Critic revision loop ----
    revision = 0
    while True:
        step("◆ Writer chain — drafting…" if revision == 0
             else f"◆ Writer chain — revising (pass {revision})…")
        report_msg = writer_chain.invoke({
             "topic": query,
            "research": research_combined,
            "today": date.today().isoformat(),
})
        state["report"] = report_msg.content          # .content = just the article

        step("◆ Critic chain — evaluating…")
        critic = critic_chain.invoke({
            "query": query,
            "research_output": state["report"],
            "sources": state["search_agent_response"],
        })
        state["critic_response"] = critic              # typed CriticResponse object

        # stop conditions
        if not critic.needs_revision:
            step(f"◆ Critic approved · score {critic.score.overall:.2f}")
            break
        if revision >= MAX_REVISIONS:
            step(f"◆ Max revisions reached ({MAX_REVISIONS}) · score {critic.score.overall:.2f}")
            break

        # feed the critic's instructions back into the research for the next writer pass
        revision += 1
        research_combined = (
            f"{research_combined}\n\n"
            f"PREVIOUS DRAFT:\n{state['report']}\n\n"
            f"REVISE THE ARTICLE ADDRESSING THESE POINTS:\n"
            + "\n".join(f"- {instr}" for instr in critic.revision_instructions)
        )

    state["revisions"] = revision
    return state


if __name__ == "__main__":
    query = "What are the latest advancements in renewable energy technologies?"

    # CLI run: print each step to the terminal
    final_state = run_research_pipline(query, on_step=print)

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print(final_state["report"])
    print(f"\n(after {final_state['revisions']} revision pass(es), "
          f"final score {final_state['critic_response'].score.overall:.2f})")