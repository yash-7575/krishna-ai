"""
utils.py
========
Helper utilities for the Krishna AI companion:
  - Japa / chanting mention detection
  - LLM-based fact extraction from conversations
  - Storing extracted facts to Chroma long-term memory
  - Fetching recent memory highlights for the sidebar
"""

import json
import re
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Japa Detection
# ---------------------------------------------------------------------------

# Keywords and patterns that indicate the user mentioned japa / chanting
_JAPA_KEYWORDS = [
    r"\bjapa\b",
    r"\bjaap\b",
    r"\brounds?\b",
    r"\bmala[s]?\b",
    r"\bmahamantra\b",
    r"\bmahamantra\b",
    r"\bhare krishna\b",
    r"\bhare ram\b",
    r"\bchanting\b",
    r"\bkirtan\b",
    r"\bnaam\s*japa\b",
    r"\bnaam\s*jaap\b",
    r"\bbeads?\b",
    r"\b\d+\s*rounds?\b",      # "16 rounds", "108 rounds"
    r"\b\d+\s*mala[s]?\b",     # "4 malas"
    r"\bsankhya\b",            # Hindi: count/number (of rounds)
]

_JAPA_PATTERN = re.compile(
    "|".join(_JAPA_KEYWORDS),
    re.IGNORECASE | re.UNICODE,
)


def detect_japa_mention(text: str) -> bool:
    """
    Returns True if the user's message mentions japa, chanting rounds,
    mahamantra, or any related spiritual practice.

    Args:
        text: The raw user message string.

    Returns:
        bool: True if japa/chanting is mentioned.
    """
    return bool(_JAPA_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Fact Extraction from Conversation Turns
# ---------------------------------------------------------------------------

_FACT_EXTRACTION_PROMPT = """You are a precise fact extractor. Analyze the conversation below between a user ("Parth") and Krishna, and extract 3 to 5 important facts or summaries about the USER (Parth) only.

Focus on:
- Personal life events (new job, health, family, relationships)
- Emotional states (happy, anxious, excited, struggling)
- Spiritual practice (japa rounds, meditation, fasting, festivals)
- Goals and aspirations
- Important decisions or milestones
- Preferences, hobbies, or interests

Rules:
- Write each fact as a concise, third-person statement about the user
- Include the date provided
- Assign a category: one of [japa, life_event, emotion, goal, health, family, spiritual, general]
- Assign importance: one of [high, medium, low]
- Return ONLY valid JSON — no extra text, no markdown fences

Output format (JSON array):
[
  {
    "fact": "User completed 16 rounds of Hare Krishna japa on {date}",
    "category": "japa",
    "importance": "high",
    "date": "{date}"
  },
  ...
]

TODAY'S DATE: {date}

USER MESSAGE:
{user_msg}

KRISHNA'S RESPONSE:
{ai_msg}

Extract facts now (JSON only):"""


async def extract_facts_from_conversation(
    llm,
    user_msg: str,
    ai_msg: str,
    date_str: str | None = None,
) -> list[dict]:
    """
    Uses the LLM to extract key facts about the user from a conversation turn.

    This is called asynchronously after every Krishna response so it doesn't
    block the streaming UI.

    Args:
        llm:      The ChatOllama instance (already initialized).
        user_msg: The user's message for this turn.
        ai_msg:   Krishna's response for this turn.
        date_str: ISO date string (defaults to today).

    Returns:
        List of dicts with keys: fact, category, importance, date.
        Returns empty list on any parsing error.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = _FACT_EXTRACTION_PROMPT.format(
        date=date_str,
        user_msg=user_msg,
        ai_msg=ai_msg,
    )

    try:
        messages = [
            SystemMessage(content="You are a JSON-only fact extraction assistant."),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip accidental markdown fences if model adds them
        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()

        # Handle <think> tags from qwen3 (reasoning tokens)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        facts = json.loads(raw)

        # Validate structure — keep only well-formed entries
        validated = []
        for item in facts:
            if isinstance(item, dict) and "fact" in item:
                validated.append({
                    "fact": str(item.get("fact", "")),
                    "category": str(item.get("category", "general")),
                    "importance": str(item.get("importance", "medium")),
                    "date": str(item.get("date", date_str)),
                })
        return validated

    except Exception as e:
        # Silently fail — memory extraction should never crash the app
        print(f"[Memory] Fact extraction failed (non-critical): {e}")
        return []


# ---------------------------------------------------------------------------
# Storing Facts to Chroma
# ---------------------------------------------------------------------------

def store_facts_to_memory(collection, facts: list[dict]) -> None:
    """
    Stores a list of extracted facts into the Chroma user_lifetime_memory
    collection. Each fact gets a unique UUID as its document ID.

    Args:
        collection: The Chroma Collection object (raw chromadb collection).
        facts:      List of fact dicts from extract_facts_from_conversation().
    """
    if not facts:
        return

    documents = []
    metadatas = []
    ids = []

    for fact in facts:
        fact_text = fact.get("fact", "").strip()
        if not fact_text:
            continue

        documents.append(fact_text)
        metadatas.append({
            "date": fact.get("date", datetime.now().strftime("%Y-%m-%d")),
            "category": fact.get("category", "general"),
            "importance": fact.get("importance", "medium"),
        })
        ids.append(str(uuid.uuid4()))

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"[Memory] Stored {len(documents)} new fact(s) about Parth.")


# ---------------------------------------------------------------------------
# Sidebar Memory Highlights
# ---------------------------------------------------------------------------

def get_recent_memory_highlights(collection, n: int = 3) -> list[str]:
    """
    Fetches the n most recently stored facts from user_lifetime_memory,
    sorted by date descending, for display in the Chainlit sidebar.

    Args:
        collection: The raw Chroma Collection object.
        n:          Number of recent facts to return.

    Returns:
        List of fact strings (most recent first).
    """
    try:
        result = collection.get(
            include=["documents", "metadatas"],
            limit=200,  # Fetch up to 200 and sort manually (Chroma limitation)
        )

        if not result or not result.get("documents"):
            return []

        # Pair documents with their date metadata for sorting
        paired = list(zip(
            result["documents"],
            result["metadatas"] or [{}] * len(result["documents"]),
        ))

        # Sort by date descending (ISO format strings sort correctly)
        paired.sort(
            key=lambda x: x[1].get("date", "0000-00-00"),
            reverse=True,
        )

        return [doc for doc, _ in paired[:n]]

    except Exception as e:
        print(f"[Memory] Could not fetch highlights (non-critical): {e}")
        return []


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------

def truncate_history(history: list, max_turns: int = 10) -> list:
    """
    Keeps only the last `max_turns` conversation turns (user + assistant pairs)
    to prevent context window overflow.

    Args:
        history:   List of LangChain message objects (HumanMessage, AIMessage).
        max_turns: Maximum number of complete turns to retain.

    Returns:
        Truncated list of messages.
    """
    # Each turn = 2 messages (human + AI), so max messages = max_turns * 2
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history
