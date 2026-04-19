"""
data_loader.py
==============
Handles:
  1. Loading the Bhagavad Gita CSV into Chroma (gita_knowledge collection)
  2. Initializing the user lifetime memory collection (user_lifetime_memory)
  3. Returning LangChain retrievers for both collections

Dataset: data/geeta_dataset.csv
Source:  https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset

The loader is idempotent — it checks if data already exists before re-loading,
so restarting the app is fast after the first run.
"""

import os
import sys
from pathlib import Path

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PATH = "./chroma_db"
GITA_COLLECTION = "gita_knowledge"
MEMORY_COLLECTION = "user_lifetime_memory"
GITA_CSV_PATH = "./data/geeta_dataset.csv"

# Minimum number of verses expected in a fully-loaded Gita collection
# (Bhagavad Gita has 700 shlokas — if we have at least 100, we're loaded)
GITA_MIN_DOCS = 100


# ---------------------------------------------------------------------------
# Column Name Mapping
# ---------------------------------------------------------------------------

# The dataset may use various column names — we normalize them here.
# Keys are our normalized names, values are lists of possible CSV column names
# (checked in order, first match wins).
COLUMN_MAP = {
    "chapter":     ["chapter", "chapter_number", "adhyay", "ch"],
    "verse":       ["verse", "verse_number", "shlok", "shloka", "sloka", "vs"],
    "sanskrit":    ["sanskrit", "shloka", "original", "devanagari", "text_devanagari"],
    "transliteration": ["transliteration", "roman", "phonetic", "text_roman"],
    "translation": ["translation", "meaning", "english", "english_meaning",
                    "text_english", "artha", "sense"],
    "explanation": ["explanation", "purport", "commentary", "description",
                    "meaning_detailed", "meaning2"],
}


def _detect_column(df: pd.DataFrame, normalized_name: str) -> str | None:
    """
    Finds the actual column name in the DataFrame for a normalized field name.
    Case-insensitive matching against the possible names list.

    Args:
        df:              The loaded DataFrame.
        normalized_name: Our internal field name (e.g., "chapter").

    Returns:
        The actual column name string, or None if not found.
    """
    candidates = COLUMN_MAP.get(normalized_name, [normalized_name])
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def _build_gita_document(row: pd.Series, col_map: dict) -> Document:
    """
    Converts a single row from the Gita CSV into a rich LangChain Document.

    The document text combines all available fields so that embedding-based
    search can find verses by topic, Sanskrit, or English meaning.

    Args:
        row:     A pandas Series (one CSV row).
        col_map: Mapping from normalized name → actual column name (or None).

    Returns:
        A LangChain Document with text content and chapter/verse metadata.
    """
    def get(field: str) -> str:
        col = col_map.get(field)
        if col and col in row.index:
            val = row[col]
            if pd.notna(val):
                return str(val).strip()
        return ""

    chapter = get("chapter")
    verse = get("verse")
    sanskrit = get("sanskrit")
    transliteration = get("transliteration")
    translation = get("translation")
    explanation = get("explanation")

    # Build the chapter.verse reference
    ref = f"{chapter}.{verse}" if chapter and verse else "?"

    # Compose the full text chunk — rich enough for semantic search
    parts = [f"Bhagavad Gita — Chapter {chapter}, Verse {verse} ({ref})"]

    if sanskrit:
        parts.append(f"Sanskrit:\n{sanskrit}")
    if transliteration:
        parts.append(f"Transliteration:\n{transliteration}")
    if translation:
        parts.append(f"Translation:\n{translation}")
    if explanation:
        parts.append(f"Purport/Explanation:\n{explanation}")

    full_text = "\n\n".join(parts)

    metadata = {
        "chapter": chapter or "unknown",
        "verse": verse or "unknown",
        "ref": ref,
        "source": "Bhagavad Gita",
    }

    return Document(page_content=full_text, metadata=metadata)


# ---------------------------------------------------------------------------
# Gita Loader
# ---------------------------------------------------------------------------

def load_gita_to_chroma(
    embeddings,
    csv_path: str = GITA_CSV_PATH,
    chroma_path: str = CHROMA_PATH,
) -> Chroma:
    """
    Loads the Bhagavad Gita CSV into the Chroma `gita_knowledge` collection.

    Idempotent: if the collection already has >= GITA_MIN_DOCS documents,
    it skips loading and returns the existing collection.

    Args:
        embeddings:  An initialized OllamaEmbeddings (or compatible) instance.
        csv_path:    Path to the Gita CSV file.
        chroma_path: Directory for the persistent Chroma DB.

    Returns:
        A LangChain Chroma vector store instance for the gita_knowledge collection.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        SystemExit:        If the CSV has no usable data.
    """
    print(f"[Gita] Initializing '{GITA_COLLECTION}' collection...")

    # Open the persistent Chroma client
    client = chromadb.PersistentClient(path=chroma_path)

    # Check if already loaded
    existing = client.get_or_create_collection(GITA_COLLECTION)
    existing_count = existing.count()

    if existing_count >= GITA_MIN_DOCS:
        print(f"[Gita] Already loaded ({existing_count} verses). Skipping re-load. ✓")
        return Chroma(
            client=client,
            collection_name=GITA_COLLECTION,
            embedding_function=embeddings,
        )

    # Validate CSV path
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n\n❌ Bhagavad Gita dataset not found at: {csv_path}\n\n"
            "Please download it:\n"
            "  mkdir -p data\n"
            "  curl -L https://huggingface.co/datasets/JDhruv14/"
            "Bhagavad-Gita_Dataset/resolve/main/geeta_dataset.csv "
            "-o data/geeta_dataset.csv\n\n"
            "See README.md for full setup instructions."
        )

    # Load the CSV
    print(f"[Gita] Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]  # Strip whitespace from headers

    print(f"[Gita] CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    if df.empty:
        print("[Gita] ❌ CSV file is empty. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Detect actual column names
    col_map = {
        field: _detect_column(df, field)
        for field in COLUMN_MAP
    }
    print(f"[Gita] Detected columns: {col_map}")

    # Build LangChain Documents
    print("[Gita] Building document chunks...")
    documents = []
    for _, row in df.iterrows():
        try:
            doc = _build_gita_document(row, col_map)
            if len(doc.page_content) > 20:  # Skip degenerate rows
                documents.append(doc)
        except Exception as e:
            print(f"[Gita] Skipping row due to error: {e}")

    if not documents:
        print("[Gita] ❌ No valid documents extracted from CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"[Gita] Built {len(documents)} verse documents. Embedding & storing...")

    # Store into Chroma in batches (avoids OOM on large datasets)
    batch_size = 50
    vector_store = None

    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                client=client,
                collection_name=GITA_COLLECTION,
            )
        else:
            vector_store.add_documents(batch)
        print(f"[Gita]   Stored verses {i + 1}–{min(i + batch_size, len(documents))}...")

    print(f"[Gita] ✅ All {len(documents)} Gita verses loaded into Chroma!")
    return vector_store


# ---------------------------------------------------------------------------
# Memory Collection Initializer
# ---------------------------------------------------------------------------

def init_memory_collection(chroma_path: str = CHROMA_PATH):
    """
    Creates or opens the `user_lifetime_memory` Chroma collection.

    Returns BOTH the raw chromadb Collection (for direct add/get operations)
    AND the LangChain Chroma wrapper (for retriever-based semantic search).

    Args:
        chroma_path: Directory for the persistent Chroma DB.

    Returns:
        Tuple of (raw_collection, langchain_chroma_instance)
        NOTE: langchain_chroma_instance does NOT have embeddings attached here —
        the embeddings are injected by get_retrievers().
    """
    client = chromadb.PersistentClient(path=chroma_path)
    raw_collection = client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity for memories
    )
    count = raw_collection.count()
    print(f"[Memory] '{MEMORY_COLLECTION}' collection ready ({count} memories stored).")
    return raw_collection, client


# ---------------------------------------------------------------------------
# Master Initializer — called once at app startup
# ---------------------------------------------------------------------------

def get_retrievers(embeddings, chroma_path: str = CHROMA_PATH):
    """
    Master initialization function. Sets up both Chroma collections and
    returns ready-to-use LangChain retrievers.

    Call this once in @cl.on_chat_start. It is safe to call multiple times
    (idempotent for the Gita collection).

    Args:
        embeddings:  An initialized OllamaEmbeddings instance.
        chroma_path: Directory for the persistent Chroma DB.

    Returns:
        dict with keys:
          - "gita_retriever":      LangChain retriever for Bhagavad Gita RAG
          - "memory_retriever":    LangChain retriever for user lifetime memories
          - "memory_collection":   Raw Chroma collection (for add/get operations)
          - "memory_chroma":       LangChain Chroma instance for memory collection
    """
    print("[Loader] ═══ Initializing Krishna AI Knowledge Base ═══")

    # ── Gita RAG collection ──────────────────────────────────────────────────
    gita_chroma = load_gita_to_chroma(embeddings, chroma_path=chroma_path)
    gita_retriever = gita_chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # Top 3 most relevant Gita verses per query
    )

    # ── User lifetime memory collection ─────────────────────────────────────
    raw_memory_collection, client = init_memory_collection(chroma_path)

    # Build LangChain Chroma wrapper for semantic retrieval on memories
    memory_chroma = Chroma(
        client=client,
        collection_name=MEMORY_COLLECTION,
        embedding_function=embeddings,
    )
    memory_retriever = memory_chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # Top 5 most relevant memories per query
    )

    print("[Loader] ✅ All collections ready. Krishna is awakening... 🙏\n")

    return {
        "gita_retriever": gita_retriever,
        "memory_retriever": memory_retriever,
        "memory_collection": raw_memory_collection,
        "memory_chroma": memory_chroma,
    }
