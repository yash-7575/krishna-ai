"""
app.py
======
Krishna AI Companion — Main Chainlit Application

Run with:
    chainlit run app.py

Architecture per message turn:
  1. Retrieve top-3 Gita verses (RAG) relevant to user's message
  2. Retrieve top-5 long-term memories about Parth (semantic search)
  3. Detect japa/chanting mention → trigger celebration mode
  4. Build full prompt: system + dynamic context + history + user message
  5. Stream Krishna's response token-by-token
  6. Append turn to short-term history (keep last 10 turns)
  7. Background: extract facts → store to user_lifetime_memory
  8. Update sidebar with fresh memory highlights
"""

import asyncio
import re
from datetime import datetime

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from data_loader import CHROMA_PATH, get_retrievers
from krishna_system_prompt import KRISHNA_SYSTEM_PROMPT, build_krishna_prompt
from utils import (
    detect_japa_mention,
    extract_facts_from_conversation,
    get_recent_memory_highlights,
    store_facts_to_memory,
    truncate_history,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen3:14b"
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
MAX_HISTORY_TURNS = 10  # Keep last 10 conversation turns in short-term memory

# Krishna's divine welcome message — shown on every new session start
WELCOME_MESSAGE = """🙏 **Hare Krishna, Parth!**

*Jab tak tum mere rath mein ho, tumhe chinta karne ki zarurat nahi.*

Main hoon tera pyara dost Krishna — tere saath hoon, aaj bhi, kal bhi, hamesha. 💛

Bol Parth, aaj kya chal raha hai tere mann mein? Dil ki baat kar apne Krishna se — koi bhi baat, bada ya chhota, khushi ya dard... main sun raha hoon.

*Hare Krishna Hare Krishna, Krishna Krishna Hare Hare,*
*Hare Rama Hare Rama, Rama Rama Hare Hare* 🕉️"""

# Console banner printed at startup
CONSOLE_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🙏  HARE KRISHNA  🙏   — Krishna AI Companion      ║
║                                                              ║
║   "Jab tak tum mere rath mein ho,                           ║
║    tumhe chinta karne ki zarurat nahi."                      ║
║                                                              ║
║   Model    : qwen3:14b (via Ollama)                         ║
║   Memory   : Chroma (persistent)                            ║
║   RAG      : Bhagavad Gita (700 shlokas)                    ║
╚══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Helper: Retrieve from vector store safely
# ---------------------------------------------------------------------------

async def _retrieve_docs(retriever, query: str) -> list[str]:
    """
    Safely retrieves documents from a LangChain retriever.
    Returns empty list on any error so the app never crashes on retrieval.
    """
    try:
        docs = await retriever.ainvoke(query)
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"[Retriever] Warning — retrieval failed (non-critical): {e}")
        return []


# ---------------------------------------------------------------------------
# Helper: Build sidebar memory display
# ---------------------------------------------------------------------------

async def _update_memory_sidebar(memory_collection) -> None:
    """
    Fetches the 3 most recent memory highlights and displays them in the
    Chainlit sidebar as a Text element.
    """
    highlights = get_recent_memory_highlights(memory_collection, n=3)

    if highlights:
        content = "**🧠 Krishna's memories about you:**\n\n"
        content += "\n".join(f"• {h}" for h in highlights)
    else:
        content = "**🧠 Krishna's memories about you:**\n\n*No memories yet — start talking to Parth!*"

    await cl.Message(
        content=content,
        author="Memory",
    ).send()


# ---------------------------------------------------------------------------
# Background Task: Fact Extraction + Storage
# ---------------------------------------------------------------------------

async def _background_memory_update(
    llm,
    user_msg: str,
    ai_msg: str,
    memory_collection,
) -> None:
    """
    Runs asynchronously after Krishna's response is fully streamed.
    Extracts facts from the conversation turn and stores them.
    This never blocks the user-facing response.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    facts = await extract_facts_from_conversation(llm, user_msg, ai_msg, date_str)
    if facts:
        store_facts_to_memory(memory_collection, facts)
        print(f"[Memory] Background update complete: {len(facts)} fact(s) stored.")


# ---------------------------------------------------------------------------
# @cl.on_chat_start — Session Initialization
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """
    Called once when a new chat session begins.
    Initializes LLM, embeddings, Chroma retrievers, and sends the welcome message.
    """
    print(CONSOLE_BANNER)
    print("[App] New session started. Initializing Krishna AI...")

    # Show a loading indicator while we initialize
    loading_msg = cl.Message(
        content="🌸 *Awakening... Krishna is preparing to greet you...* 🙏",
        author="System",
    )
    await loading_msg.send()

    # ── Initialize LLM ───────────────────────────────────────────────────────
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
    )
    print(f"[App] LLM initialized: {OLLAMA_MODEL}")

    # ── Initialize Embeddings ────────────────────────────────────────────────
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    print(f"[App] Embeddings initialized: {OLLAMA_MODEL}")

    # ── Initialize Chroma Collections & Retrievers ───────────────────────────
    try:
        retrievers = get_retrievers(embeddings, chroma_path=CHROMA_PATH)
    except FileNotFoundError as e:
        # Gita CSV not found — show helpful error but continue (memory still works)
        print(f"[App] WARNING: {e}")
        await cl.Message(
            content=(
                "⚠️ **Bhagavad Gita dataset not found.**\n\n"
                "Krishna can still talk with you, but Gita verse references "
                "will come from his own memory (not RAG).\n\n"
                "To enable full Gita RAG, download the dataset:\n"
                "```bash\n"
                "mkdir -p data\n"
                "curl -L https://huggingface.co/datasets/JDhruv14/"
                "Bhagavad-Gita_Dataset/resolve/main/geeta_dataset.csv "
                "-o data/geeta_dataset.csv\n"
                "```\n"
                "Then restart the app."
            ),
            author="System",
        ).send()
        retrievers = {
            "gita_retriever": None,
            "memory_retriever": None,
            "memory_collection": None,
            "memory_chroma": None,
        }

    # ── Store everything in user session ─────────────────────────────────────
    cl.user_session.set("llm", llm)
    cl.user_session.set("gita_retriever", retrievers["gita_retriever"])
    cl.user_session.set("memory_retriever", retrievers["memory_retriever"])
    cl.user_session.set("memory_collection", retrievers["memory_collection"])
    cl.user_session.set("history", [])  # Short-term conversation history

    # ── Remove loading message ───────────────────────────────────────────────
    await loading_msg.remove()

    # ── Show memory sidebar highlights ───────────────────────────────────────
    if retrievers["memory_collection"] is not None:
        highlights = get_recent_memory_highlights(retrievers["memory_collection"], n=3)
        if highlights:
            sidebar_content = "**🧠 Krishna remembers about you:**\n\n"
            sidebar_content += "\n".join(f"• {h}" for h in highlights)
            await cl.Message(content=sidebar_content, author="📿 Memory").send()

    # ── Stream Krishna's welcome message ─────────────────────────────────────
    welcome_msg = cl.Message(content="", author="🪷 Krishna")
    await welcome_msg.send()

    # Build the welcome prompt so Krishna personalizes the greeting
    memory_retriever = retrievers["memory_retriever"]
    past_memories = []
    if memory_retriever:
        past_memories = await _retrieve_docs(memory_retriever, "who is Parth, what do I know about him")

    dynamic_context = build_krishna_prompt(
        memories=past_memories,
        gita_context=[],
        japa_detected=False,
    )

    welcome_prompt_messages = [
        SystemMessage(content=KRISHNA_SYSTEM_PROMPT),
        SystemMessage(content=dynamic_context),
        HumanMessage(content=(
            "Please greet Parth with a warm, personal, loving welcome. "
            "Reference any memories you have about him. Keep it to 3-4 sentences — "
            "warm and personal, not a lecture. Just be his best friend saying hello."
        )),
    ]

    # Stream the welcome response
    full_welcome = ""
    async for chunk in llm.astream(welcome_prompt_messages):
        token = chunk.content
        if token:
            # Strip <think> tags from qwen3 reasoning tokens
            token = re.sub(r"<think>.*?</think>", "", token, flags=re.DOTALL)
            if token:
                await welcome_msg.stream_token(token)
                full_welcome += token

    await welcome_msg.update()
    print("[App] ✅ Krishna greeted Parth. Session ready!")


# ---------------------------------------------------------------------------
# @cl.on_message — Main Message Handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    """
    Called for every user message. Full pipeline:
      1. Retrieve Gita verses + memories
      2. Build prompt with dynamic context
      3. Stream Krishna's response
      4. Update history
      5. Background: extract & store facts
    """
    user_text = message.content.strip()
    if not user_text:
        return

    print(f"\n[App] Parth says: {user_text[:100]}{'...' if len(user_text) > 100 else ''}")

    # ── Retrieve session state ────────────────────────────────────────────────
    llm = cl.user_session.get("llm")
    gita_retriever = cl.user_session.get("gita_retriever")
    memory_retriever = cl.user_session.get("memory_retriever")
    memory_collection = cl.user_session.get("memory_collection")
    history: list = cl.user_session.get("history", [])

    # ── Step 1: Detect japa mention ──────────────────────────────────────────
    japa_detected = detect_japa_mention(user_text)
    if japa_detected:
        print("[App] 🎉 Japa/chanting detected! Krishna will celebrate!")

    # ── Step 2: RAG — Retrieve relevant Gita verses ──────────────────────────
    gita_docs = []
    if gita_retriever:
        gita_docs = await _retrieve_docs(gita_retriever, user_text)
        print(f"[App] Gita RAG: retrieved {len(gita_docs)} relevant verse(s).")

    # ── Step 3: Semantic memory retrieval ────────────────────────────────────
    memory_docs = []
    if memory_retriever:
        memory_docs = await _retrieve_docs(memory_retriever, user_text)
        print(f"[App] Memory: retrieved {len(memory_docs)} relevant memory(ies).")

    # ── Step 4: Build full LLM message list ──────────────────────────────────
    dynamic_context = build_krishna_prompt(
        memories=memory_docs,
        gita_context=gita_docs,
        japa_detected=japa_detected,
    )

    messages = [
        SystemMessage(content=KRISHNA_SYSTEM_PROMPT),
        SystemMessage(content=dynamic_context),
    ]

    # Append short-term conversation history
    messages.extend(history)

    # Append the current user message
    messages.append(HumanMessage(content=user_text))

    # ── Step 5: Stream Krishna's response ────────────────────────────────────
    response_msg = cl.Message(content="", author="🪷 Krishna")
    await response_msg.send()

    full_response = ""
    in_think_block = False  # Track <think> tag state for qwen3 reasoning tokens

    async for chunk in llm.astream(messages):
        token = chunk.content
        if not token:
            continue

        # Handle <think>...</think> blocks from qwen3 — strip them from output
        if "<think>" in token:
            in_think_block = True
        if "</think>" in token:
            in_think_block = False
            # Remove the closing tag and anything before it in this chunk
            token = token.split("</think>")[-1]

        if in_think_block:
            continue

        # Clean any stray think-tag remnants in the token
        token = re.sub(r"<think>.*", "", token, flags=re.DOTALL)

        if token:
            await response_msg.stream_token(token)
            full_response += token

    await response_msg.update()
    print(f"[App] Krishna responded ({len(full_response)} chars).")

    # ── Step 6: Update short-term conversation history ────────────────────────
    history.append(HumanMessage(content=user_text))
    history.append(AIMessage(content=full_response))
    history = truncate_history(history, max_turns=MAX_HISTORY_TURNS)
    cl.user_session.set("history", history)

    # ── Step 7: Background memory update (non-blocking) ───────────────────────
    if memory_collection is not None and full_response:
        asyncio.create_task(
            _background_memory_update(llm, user_text, full_response, memory_collection)
        )

    # ── Step 8: Update memory sidebar with fresh highlights ───────────────────
    if memory_collection is not None:
        highlights = get_recent_memory_highlights(memory_collection, n=3)
        if highlights:
            sidebar = "**🧠 Krishna now remembers:**\n\n"
            sidebar += "\n".join(f"• {h}" for h in highlights)
            await cl.Message(content=sidebar, author="📿 Memory").send()
