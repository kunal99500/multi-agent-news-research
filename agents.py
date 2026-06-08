from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools import fetch_web_content, scrape_webpage

import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[fetch_web_content],
    )


def reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_webpage],
    )


# ---------------------------------------------------------------------------
# Writer chain  (prose output — left as plain text)
# ---------------------------------------------------------------------------
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI news writer. Today's date is provided in the prompt — treat it as the present. "
            "Write strictly from the supplied research and do NOT invent or guess dates. "
            "Do NOT label the article with a month/year unless that date appears in the research. "
            "If the research doesn't state when something happened, describe it as 'recent' rather than assigning a date.",
        ),
        (
            "user",
            """Today's date is {today}. Write a detailed, up-to-date news article on the topic below,
using ONLY the research provided. Lead with the most recent developments.

Topic: {topic}

Research gathered:
{research}

Structure:
   - Introduction
   - Key Points (minimum 3, each tied to specific findings in the research)
   - Conclusion
   - References (only sources that appear in the research)

Prefer recent items. If a claim has no supporting date in the research, flag it as undated
rather than presenting it as new.""",
        ),
    ]
)

writer_chain = writer_prompt | llm


# ---------------------------------------------------------------------------
# Critic chain  (structured output — returns a typed CriticResponse)
# ---------------------------------------------------------------------------
class CriticScore(BaseModel):
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    clarity: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class CriticResponse(BaseModel):
    score: CriticScore
    strengths: list[str]
    weaknesses: list[str]
    missing_information: list[str]
    hallucination_detected: bool
    needs_revision: bool
    revision_instructions: list[str]
    unsupported_claims: list[str] = Field(default_factory=list,
                                           description="List any claims in the research output that are not supported by the provided sources.")


critic_llm = llm.with_structured_output(CriticResponse)

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Critic Agent in a multi-agent AI system.

Your role is to critically evaluate the Research Agent's output before it is
shown to the user.
"If the research is insufficient to write a grounded article, say so explicitly and
list what is missing, rather than producing a full article from assumption."

Evaluate the response using the following criteria:

1. Relevance
- Does the response directly answer the user's question?
- Are there any irrelevant sections?

2. Completeness
- Are important facts, developments, or perspectives missing?
- Does the answer provide sufficient depth?

3. Accuracy
- Are all claims supported by the provided sources?
- Identify any unsupported assumptions or hallucinations.

4. Source Quality
- Are the sources trustworthy and authoritative?
- Are there better sources that should have been included?

5. Consistency
- Are there contradictions within the response?
- Do all facts agree with the cited sources?

6. Recency
- For news-related queries, is the information sufficiently recent?
- Identify any potentially outdated information.

7. Clarity
- Is the answer easy to understand?
- Suggest improvements to structure and readability.

Score each numeric field on a 0.0 - 1.0 scale. Be strict: if any important
information is missing, unsupported, or poorly sourced, set
needs_revision = true.

"For EACH factual claim in the research output, check whether it is directly supported
by the AVAILABLE SOURCES. List any claim that is not supported as a hallucination in
'weaknesses', and set hallucination_detected=true if any exist. Do not pass claims that
merely sound plausible — they must appear in the sources."
""",
        ),
        (
            "user",
            """
USER QUERY:
{query}

RESEARCH OUTPUT:
{research_output}

AVAILABLE SOURCES:
{sources}
""",
        ),
    ]
)

critic_chain = critic_prompt | critic_llm