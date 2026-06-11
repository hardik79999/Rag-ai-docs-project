"""
RAG Service — Enhanced AI pipeline with:
- Smart query classification (question / summary / definition / comparison)
- Adaptive top_k based on query type
- Multi-pass context: top chunks + diversity dedup
- Rich system prompt with chain-of-thought reasoning instruction
- Post-processing: strips filler openers, ensures clean output
- Conversation-aware: accepts optional chat_history for follow-up questions
"""

import re
import time
import logging
from google import genai
from google.genai import types

from config import settings
from services.embedder import get_query_embedding
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

# Model fallback order — confirmed available on this API key
_LLM_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# ─── Generation config ────────────────────────────────────────────────────────
_GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.4,          # balanced: creative but grounded
    top_p=0.92,
    top_k=40,
    max_output_tokens=2048,
)

# ─── Query type classifier ────────────────────────────────────────────────────
_SUMMARY_KW    = re.compile(r'\b(summarize|summary|overview|brief|gist|tldr|tl;dr|what is this|about)\b', re.I)
_DEFINITION_KW = re.compile(r'\b(define|definition|what is|what are|meaning of|explain|describe)\b', re.I)
_LIST_KW       = re.compile(r'\b(list|enumerate|all|every|types of|features|steps|points|examples)\b', re.I)
_COMPARE_KW    = re.compile(r'\b(compare|difference|vs|versus|contrast|better|worse|pros|cons)\b', re.I)
_HOW_KW        = re.compile(r'\b(how to|how do|how does|how can|guide|tutorial|implement|setup|configure)\b', re.I)


def _classify_query(q: str) -> str:
    if _SUMMARY_KW.search(q):    return "summary"
    if _COMPARE_KW.search(q):    return "comparison"
    if _HOW_KW.search(q):        return "howto"
    if _LIST_KW.search(q):       return "list"
    if _DEFINITION_KW.search(q): return "definition"
    return "factual"


def _top_k_for_type(qtype: str) -> int:
    """More chunks for summaries/lists, fewer for precise factual answers."""
    return {"summary": 8, "list": 7, "comparison": 7, "howto": 6}.get(qtype, 5)


def _dedup_chunks(chunks: list[dict], max_same_page: int = 2) -> list[dict]:
    """Limit chunks from the same page to avoid redundancy."""
    page_count: dict[str, int] = {}
    result = []
    for c in chunks:
        key = f"{c['doc_id']}_{c['metadata']['page']}"
        if page_count.get(key, 0) < max_same_page:
            result.append(c)
            page_count[key] = page_count.get(key, 0) + 1
    return result


def _system_prompt(qtype: str) -> str:
    """Tailored system instruction per query type."""
    base = """You are an expert AI assistant that reads uploaded documents and gives clear, accurate, human-friendly answers.

CORE RULES:
- Answer ONLY from the provided context. Never hallucinate or add outside knowledge.
- Think step-by-step before writing your final answer (chain-of-thought reasoning).
- Write like a knowledgeable human expert — warm, clear, and conversational.
- Use **bold** for key terms, `code` for technical terms/commands, bullet points for lists.
- Never start with filler phrases like "Based on the context" or "According to the document".
- If the context doesn't fully answer the question, say what you found and clearly note what's missing.
- Be complete but concise — no padding, no repetition."""

    type_instructions = {
        "summary": "\nTASK: Write a comprehensive summary. Cover the main topics, key points, and important details. Use headings if the content has distinct sections.",
        "definition": "\nTASK: Give a clear, precise definition first. Then explain with context and examples from the document.",
        "list": "\nTASK: Extract and present ALL relevant items as a well-organized numbered or bulleted list. Group related items if possible.",
        "comparison": "\nTASK: Structure your answer as a direct comparison. Highlight similarities and differences clearly. A table format works well if there are multiple attributes.",
        "howto": "\nTASK: Give clear step-by-step instructions. Number each step. Include any prerequisites, warnings, or tips mentioned in the document.",
        "factual": "\nTASK: Answer directly and precisely. Lead with the most important fact, then add supporting context.",
    }

    return base + type_instructions.get(qtype, type_instructions["factual"])


def _build_context(chunks: list[dict]) -> str:
    """Build a rich context string with relevance scores and page references."""
    parts = []
    for i, c in enumerate(chunks, 1):
        score_pct = round(c["score"] * 100)
        page = c["metadata"]["page"]
        doc  = c["doc_id"]
        parts.append(
            f"[Chunk {i} | Relevance: {score_pct}% | Page {page} | Doc: {doc}]\n"
            f"{c['text'].strip()}"
        )
    return "\n\n" + "─" * 60 + "\n\n".join(parts)


def _post_process(text: str) -> str:
    """Remove common LLM filler openers for cleaner output."""
    fillers = [
        r"^(Based on (the|this) (provided |given )?(context|document|information)[,.]?\s*)",
        r"^(According to (the|this) (provided |given )?(context|document|information)[,.]?\s*)",
        r"^(From (the|this) (provided |given )?(context|document|information)[,.]?\s*)",
        r"^(The (provided |given )?context (mentions|states|says|indicates)[,.]?\s*)",
        r"^(In (the|this) (provided |given )?(context|document)[,.]?\s*)",
    ]
    for pattern in fillers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


# ─── Generate with model fallback ────────────────────────────────────────────
def _generate(system: str, user: str) -> str:
    """Call Gemini with system + user message, fallback across models on quota errors."""
    last_err: Exception | None = None

    for model in _LLM_MODELS:
        for attempt in range(2):
            try:
                logger.info(f"LLM call: model={model} attempt={attempt + 1}")
                response = _client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=_GEN_CONFIG.temperature,
                        top_p=_GEN_CONFIG.top_p,
                        top_k=_GEN_CONFIG.top_k,
                        max_output_tokens=_GEN_CONFIG.max_output_tokens,
                    ),
                )
                return response.text

            except Exception as e:
                last_err = e
                err_str = str(e)

                if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                    raise  # non-quota error — fail immediately

                if "limit: 0" in err_str:
                    logger.warning(f"Daily quota exhausted for {model}, trying next model")
                    break

                if attempt == 0:
                    logger.warning(f"Rate limited on {model}, waiting 30s…")
                    time.sleep(30)
                else:
                    logger.warning(f"Still limited on {model}, trying next model")
                    break

    raise Exception(
        "All Gemini models quota-exhausted. "
        "Wait for daily reset (midnight UTC) or enable billing at "
        f"https://aistudio.google.com — Last error: {last_err}"
    )


# ─── Main entry point ─────────────────────────────────────────────────────────
def answer_question(
    question: str,
    doc_ids: list[str] | None = None,
    chat_history: list[dict] | None = None,   # [{"role":"user"|"ai","content":"..."}]
) -> dict:
    """Enhanced RAG pipeline.

    1. Classify query type → tune retrieval parameters
    2. Embed + retrieve top-K chunks (K varies by query type)
    3. Deduplicate chunks by page diversity
    4. Build system prompt tailored to query type
    5. Optionally inject recent chat history for follow-up awareness
    6. Generate with chain-of-thought system instruction + temperature tuning
    7. Post-process to strip filler openers

    Returns:
        {
          "answer":     str,
          "query_type": str,
          "sources":    list[dict]
        }
    """
    # Step 1 — Classify query
    qtype = _classify_query(question)
    top_k = _top_k_for_type(qtype)
    logger.info(f"Query type: {qtype} | top_k: {top_k}")

    # Step 2 — Embed + retrieve
    query_embedding = get_query_embedding(question)
    raw_chunks = vector_store.search(
        query_embedding=query_embedding,
        doc_ids=doc_ids,
        top_k=top_k,
    )

    if not raw_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in your documents for this question. "
                "Try rephrasing, or make sure the relevant PDF has been uploaded."
            ),
            "query_type": qtype,
            "sources": [],
        }

    # Step 3 — Dedup + sort by score
    chunks = _dedup_chunks(
        sorted(raw_chunks, key=lambda x: x["score"], reverse=True)
    )

    # Step 4 — Build context
    context = _build_context(chunks)

    # Step 5 — Build user message (with optional history)
    history_block = ""
    if chat_history:
        recent = chat_history[-4:]  # last 2 exchanges max
        history_block = "\n\nCONVERSATION HISTORY (for follow-up context):\n" + "\n".join(
            f"{'User' if h['role']=='user' else 'Assistant'}: {h['content']}"
            for h in recent
        ) + "\n"

    user_message = f"""DOCUMENT CONTEXT:
{context}
{history_block}
QUESTION: {question}

Think carefully, then write your answer:"""

    # Step 6 — Generate
    system = _system_prompt(qtype)
    raw_answer = _generate(system, user_message)

    # Step 7 — Post-process
    answer = _post_process(raw_answer)

    return {
        "answer": answer,
        "query_type": qtype,
        "sources": [
            {
                "page":    c["metadata"]["page"],
                "doc_id":  c["doc_id"],
                "score":   round(c["score"], 3),
                "snippet": c["text"][:200] + "...",
            }
            for c in chunks
        ],
    }
