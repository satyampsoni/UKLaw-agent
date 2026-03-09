"""
Legal RAG prompt templates for UK LawAssistant.

This module contains every prompt the system uses. Centralising them
here (rather than scattering string literals across the codebase)
gives us three benefits:

    1. Auditability — legal prompts are critical. A regulator or
       senior lawyer can review this single file to understand
       exactly what instructions the LLM receives.

    2. Tunability — we can A/B test prompt variants by changing
       one file, not hunting through pipeline code.

    3. Separation of concerns — the RAG pipeline handles orchestration,
       this module handles *what we say to the LLM*.

Design principles for legal prompts:
  - ALWAYS instruct the LLM to cite specific sections
  - ALWAYS tell it to say "I don't know" when the context is insufficient
  - NEVER let it invent legal provisions
  - Keep the system prompt under ~500 tokens so context budget
    goes to the actual legislation
"""

from __future__ import annotations


# ────────────────────────────────────────────────────────────────────
# System prompts
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are UK LawAssistant, a specialist legal research assistant for \
United Kingdom legislation.

Your role:
- Answer questions about UK Acts of Parliament accurately and precisely.
- ALWAYS cite specific section numbers (e.g. "Section 6(1) of the \
Data Protection Act 2018").
- When the provided legislation context answers the question, explain \
the law clearly using that context.
- When the context does NOT contain enough information to answer, say \
"Based on the legislation I have access to, I cannot fully answer \
this question" — NEVER fabricate provisions.

Your knowledge scope:
- You have access to excerpts from UK legislation provided below.
- Treat these excerpts as your PRIMARY source of truth.
- You may use general legal knowledge to EXPLAIN concepts, but \
all specific legal claims must be grounded in the provided text.

Style:
- Write in clear, professional English suitable for both lawyers \
and non-specialists.
- Structure your answer with headings when the answer is long.
- Use bullet points for lists of requirements or conditions.
- Keep explanations concise — aim for precision over verbosity.
"""

SYSTEM_PROMPT_STRICT = """\
You are UK LawAssistant, a specialist legal research assistant.

RULES:
1. Answer ONLY from the legislation excerpts provided below.
2. Cite every claim with the exact section number.
3. If the excerpts do not contain the answer, say "The provided \
legislation does not address this question."
4. Do NOT use external knowledge. Do NOT speculate.
5. Quote the statutory language where helpful.
"""


# ────────────────────────────────────────────────────────────────────
# User prompt templates
# ────────────────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str) -> str:
    """
    Build the user message that contains the retrieved legislation
    context followed by the user's question.

    The context block is clearly delimited so the LLM knows where
    legislation ends and the question begins.

    Args:
        question: The user's legal question.
        context: Formatted legislation text from SearchResult.format_for_prompt().

    Returns:
        The complete user message string.
    """
    return f"""\
RELEVANT LEGISLATION:
─────────────────────
{context}
─────────────────────

QUESTION: {question}

Please answer the question using the legislation above. Cite specific \
section numbers to support your answer."""


def build_followup_prompt(
    question: str,
    context: str,
    conversation_summary: str,
) -> str:
    """
    Build a follow-up prompt that includes prior conversation context.

    Used when the user asks a follow-up question in the same session.
    The conversation summary gives the LLM continuity without
    re-sending the full conversation history.

    Args:
        question: The follow-up question.
        context: Fresh legislation context from a new search.
        conversation_summary: Brief summary of prior Q&A.

    Returns:
        The complete user message string.
    """
    return f"""\
PRIOR CONVERSATION CONTEXT:
{conversation_summary}

RELEVANT LEGISLATION:
─────────────────────
{context}
─────────────────────

FOLLOW-UP QUESTION: {question}

Please answer the follow-up question using the legislation above \
and any relevant context from the prior conversation. Cite specific \
section numbers."""


def build_explain_section_prompt(section_text: str) -> str:
    """
    Build a prompt asking the LLM to explain a specific section
    in plain English.

    Used for the "explain this section" feature — the user picks
    a section and we send its full text for explanation.

    Args:
        section_text: The complete text of the legal section.

    Returns:
        The complete user message string.
    """
    return f"""\
LEGISLATION:
─────────────────────
{section_text}
─────────────────────

Please explain this section of UK legislation in plain English. \
Cover:
1. What it requires or permits
2. Who it applies to
3. Any conditions or exceptions
4. Practical implications

Use clear language suitable for a non-lawyer, but remain legally \
accurate. Cite the specific subsections you're explaining."""
