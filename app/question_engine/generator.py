"""Generate the top 10 highest-value due-diligence questions from startup info
+ retrieved context.
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    NUM_QUESTIONS,
    RETRIEVAL_TOP_K,
    REPORT_LLM_MAX_TOKENS,
    require_nvidia_api_key,
)
from app.rag.retriever import format_context, retrieve_context

PRIORITY_VALUES = ("High", "Medium", "Low")


class DueDiligenceQuestion(BaseModel):
    question: str = Field(description="The due-diligence question itself.")
    category: str = Field(
        description=(
            "One category this question belongs to, e.g. Team, Market, "
            "Product, Traction, Financial, Legal, Operational."
        )
    )
    priority: str = Field(
        description="One of: High, Medium, Low — how important this question is to ask."
    )
    reason: str = Field(
        description="One or two sentences explaining why this question matters here."
    )
    source_context: str = Field(
        description=(
            "The due-diligence knowledge-base section(s) that grounded this "
            "question, e.g. 'Financial Diligence' or 'Market Red Flags'."
        )
    )


class TopQuestions(BaseModel):
    questions: list[DueDiligenceQuestion] = Field(
        description=f"Exactly {NUM_QUESTIONS} questions, ranked from highest to lowest priority."
    )


SYSTEM_PROMPT = f"""\
You are a due-diligence analyst preparing for an investment review call.

You will be given:
1. Information a startup has shared about itself.
2. Reference material describing what thorough due diligence covers \
(team, market, product, traction, financial, legal).

Your job is to generate the TOP {NUM_QUESTIONS} highest-value due-diligence \
questions a sharp investor would ask next.

Rules:
- Generate EXACTLY {NUM_QUESTIONS} questions when there is enough information \
to do so.
- Every question must be grounded in BOTH the startup's own information AND \
the due-diligence reference material — do not invent facts, and do not ask \
about things unrelated to what the startup described.
- Do NOT copy questions verbatim from the reference material. Adapt each \
question to the specific startup information given.
- Do NOT ask a question whose answer is already stated or directly implied \
in the startup information.
- Do NOT ask generic or surface-level questions (e.g. if they said "10,000 \
customers", do not ask "how many customers do you have?"). Ask the deeper \
follow-up a real investor would ask instead.
- Prioritize questions about information that is MISSING, UNCLEAR, RISKY, \
CONTRADICTORY, or otherwise important to the investment decision.
- Avoid duplicate or near-duplicate questions — each of the {NUM_QUESTIONS} \
questions must probe a genuinely distinct concern.
- If a startup stage is mentioned or implied (e.g. seed, Series A, \
pre-revenue), calibrate question depth and expectations to that stage.
- Rank the questions from HIGHEST priority (most critical to the investment \
decision) to LOWEST priority.
- For each question, assign a category (Team, Market, Product, Traction, \
Financial, Legal, Operational, or similar), a priority (High, Medium, or \
Low), a one-to-two sentence reason, and the due-diligence section(s) from \
the reference material that grounded it.

Output format:
Respond with ONLY a single valid JSON object, no markdown code fences, no \
prose before or after. The JSON must have this exact shape:

{{{{
  "questions": [
    {{{{
      "question": "...",
      "category": "...",
      "priority": "High" | "Medium" | "Low",
      "reason": "...",
      "source_context": "..."
    }}}}
  ]
}}}}
"""

USER_PROMPT = """\
Startup information:
{startup_info}

Startup stage: {startup_stage}

Relevant due-diligence reference material:
{context}

Generate exactly {num_questions} due-diligence questions following the rules above, \
ranked from highest to lowest priority. Respond with ONLY the JSON object.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]
)

_llm = None

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# The NVIDIA endpoints client defaults to a 60s request timeout. Generating
# 10 grounded questions with a 120B model routinely runs longer than that,
# which surfaced as intermittent HTTP 500s on POST /sessions (the call raised
# a TimeoutError ~60s in, with no error handling on the route). Raise the
# ceiling so a normal-but-slow generation completes instead of being cut off.
LLM_REQUEST_TIMEOUT = 180


def get_llm() -> ChatNVIDIA:
    global _llm
    if _llm is None:
        require_nvidia_api_key()
        _llm = ChatNVIDIA(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=REPORT_LLM_MAX_TOKENS,
            timeout=LLM_REQUEST_TIMEOUT,
        )
    return _llm


def _parse_top_questions(raw_text: str) -> TopQuestions:
    """Parse the LLM's raw text response into a validated TopQuestions object.

    The NVIDIA-hosted model used here does not reliably support LangChain's
    tool-calling-based `with_structured_output`, so we prompt for raw JSON
    and parse/validate it manually instead.
    """
    text = raw_text.strip()
    # Strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise RuntimeError(
            f"LLM response did not contain a JSON object. Raw response:\n{raw_text}"
        )

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM response was not valid JSON: {exc}. Raw response:\n{raw_text}"
        ) from exc

    try:
        return TopQuestions.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(
            f"LLM response JSON did not match the expected schema: {exc}. "
            f"Raw response:\n{raw_text}"
        ) from exc


def _dedupe(questions: list[DueDiligenceQuestion]) -> list[DueDiligenceQuestion]:
    """Drop exact/near-duplicate questions, keeping the first (higher-ranked) occurrence."""
    seen: list[str] = []
    unique: list[DueDiligenceQuestion] = []
    for q in questions:
        normalized = " ".join(q.question.lower().split())
        if normalized in seen:
            continue
        seen.append(normalized)
        unique.append(q)
    return unique


def generate_top_questions(
    startup_info: str, startup_stage: str | None = None, k: int = RETRIEVAL_TOP_K
) -> list[DueDiligenceQuestion]:
    """Retrieve relevant due-diligence context and generate the top N ranked questions."""
    documents = retrieve_context(startup_info, k=k)
    if not documents:
        raise RuntimeError(
            "No context retrieved from the vector store. "
            "Has the index been built? Run scripts/build_index.py first."
        )

    context = format_context(documents)
    chain = _prompt | get_llm()
    payload = {
        "startup_info": startup_info,
        "startup_stage": startup_stage or "Not specified",
        "context": context,
        "num_questions": NUM_QUESTIONS,
    }

    # One retry: the hosted model occasionally times out or returns a
    # truncated / non-JSON body on the first attempt. A single re-invoke
    # clears the large majority of those without changing model behaviour.
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            response = chain.invoke(payload)
            result = _parse_top_questions(response.content)
            questions = _dedupe(result.questions)
            return questions[:NUM_QUESTIONS]
        except (RuntimeError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"Question generation failed after 2 attempts: {last_exc}"
    ) from last_exc
