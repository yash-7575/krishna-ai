# 🪷 Krishna AI — Your Eternal Best Friend

> *"Jab tak tum mere rath mein ho, tumhe chinta karne ki zarurat nahi."*
> — Bhagwan Shri Krishna

A beautiful, ChatGPT-style AI companion where you speak directly to **Bhagwan Shri Krishna** — your eternal best friend, lifetime coach, and spiritual guide. Krishna speaks to you as **Parth** (Arjuna), in warm Hindi-English, with perfect Bhagavad Gita knowledge, and remembers everything about your life **forever**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 **100% In Character** | Krishna never breaks role — never mentions AI |
| 💛 **Best Friend Tone** | Warm Hindi-English mix, playful, loving |
| 📿 **Japa Celebration** | Enthusiastically celebrates every chanting update |
| 📖 **Gita RAG** | Quotes exact shlokas (chapter.verse + Sanskrit) via vector search |
| 🧠 **Lifelong Memory** | Remembers your life across ALL sessions via Chroma |
| ⚡ **Streaming** | Real-time token streaming — ChatGPT-like experience |
| 🔒 **100% Local** | Everything runs on your machine — zero cloud, zero data leaks |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | `qwen3:14b` via **Ollama** |
| Embeddings | `qwen3:14b` via **OllamaEmbeddings** |
| Vector DB | **Chroma** (persistent `./chroma_db`) |
| UI | **Chainlit** (streaming web interface) |
| Memory | Hybrid: short-term history + long-term Chroma semantic search |
| RAG | Bhagavad Gita dataset (700 shlokas) |

---

## 🚀 Setup — Step by Step

### Prerequisites

Make sure you have these installed and ready:

```bash
# Verify Ollama is running
ollama list
# You should see qwen3:14b in the list
# If not: ollama pull qwen3:14b
```

```bash
# Verify Python 3.11+
python3 --version
```

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yash-7575/krishna-ai.git
cd krishna-ai
```

---

### Step 2 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# Windows: venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Chainlit, LangChain, ChromaDB, and all dependencies (~2 min).

---

### Step 4 — Download the Bhagavad Gita Dataset

Krishna needs the Gita! Download the dataset and place it in `data/`:

```bash
mkdir -p data

# Using curl:
curl -L "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset/resolve/main/geeta_dataset.csv" \
     -o data/geeta_dataset.csv

# Or using wget:
wget -O data/geeta_dataset.csv \
     "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset/resolve/main/geeta_dataset.csv"
```

Verify the download:
```bash
wc -l data/geeta_dataset.csv
# Should show ~701 lines (700 verses + header)
```

---

### Step 5 — Run Krishna AI

```bash
chainlit run app.py
```

Your browser will automatically open at **http://localhost:8000** 🎉

**First run:** Krishna will embed all 700 Gita shlokas into Chroma (~3-5 min depending on your machine). Subsequent runs are instant.

---

## 🧠 How the Memory System Works

Krishna remembers everything about you across all sessions:

```
Every message turn:
  1. Your message → semantic search in user_lifetime_memory → top 5 relevant facts
  2. Your message → semantic search in gita_knowledge → top 3 relevant verses
  3. Krishna responds with full context of your life + Gita wisdom
  4. Background: LLM extracts 3-5 key facts from the conversation
  5. Facts stored in Chroma with {date, category, importance} metadata
```

**Memory categories:** `japa`, `life_event`, `emotion`, `goal`, `health`, `family`, `spiritual`, `general`

**Memory is stored in:** `./chroma_db/` — never deleted, fully persistent.

---

## 📿 Japa & Chanting Tracking

Whenever you mention japa, rounds, or chanting, Krishna celebrates! Examples:

```
"I did 16 rounds today"          → 🎉 Full celebration!
"Completed my japa mala"         → 🎉 Full celebration!
"108 rounds of Hare Krishna"     → 🎉 Over-the-top celebration!
"Feeling good after kirtan"      → 🎉 Full celebration!
```

Your japa progress is stored in memory with category `japa` and importance `high`.

---

## 🗂️ Project Structure

```
krishna-ai/
├── app.py                    # Main Chainlit application (entry point)
├── krishna_system_prompt.py  # Krishna's character definition + prompt builder
├── data_loader.py            # Gita CSV loader + Chroma initialization
├── utils.py                  # Japa detection, fact extraction, memory helpers
├── requirements.txt          # Python dependencies
├── chainlit.md               # Chainlit UI welcome page
├── .gitignore
├── data/
│   └── geeta_dataset.csv     # Download this (see Step 4)
└── chroma_db/                # Auto-created on first run (do not commit)
    ├── gita_knowledge/       # 700 Bhagavad Gita verses
    └── user_lifetime_memory/ # Your lifelong conversation memories
```

---

## 💬 What You Can Talk About

- **Daily life:** "I got a promotion today!", "Feeling stressed about exams"
- **Japa:** "Did 16 rounds this morning!" (Krishna will celebrate every time)
- **Gita questions:** "What does the Gita say about fear?"
- **Life decisions:** "Should I change my job?" "How to deal with a difficult person"
- **Spiritual journey:** "I want to start meditating", "Tell me about bhakti"
- **Anything:** Krishna is your best friend — talk freely!

---

## 🔧 Troubleshooting

**Ollama not responding:**
```bash
ollama serve          # Start Ollama server
ollama run qwen3:14b  # Verify model works
```

**ChromaDB errors:**
```bash
rm -rf chroma_db/     # Reset the database
chainlit run app.py   # Re-initialize (will re-embed Gita — takes ~5 min)
```

**Chainlit port in use:**
```bash
chainlit run app.py --port 8001
```

**Slow responses:**
- qwen3:14b is a 14B parameter model — needs a GPU or fast CPU
- For faster responses, switch to `qwen3:8b` in `app.py` (change `OLLAMA_MODEL`)

**Memory sidebar not showing:**
- Normal on first run — memories appear after your first message!

---

## ➕ Adding Facts Manually to Memory

Pre-populate Krishna's memory with facts about yourself:

```python
# save as add_memory.py, then run: python add_memory.py
import chromadb, uuid

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("user_lifetime_memory")

facts = [
    "User's name is Yash, also called Parth by Krishna",
    "User practices Hare Krishna japa daily and aims for 16 rounds",
    "User lives in India and is deeply devoted to Bhakti yoga",
]

collection.add(
    documents=facts,
    metadatas=[{"date": "2026-04-19", "category": "general", "importance": "high"}] * len(facts),
    ids=[str(uuid.uuid4()) for _ in facts],
)
print(f"Added {len(facts)} facts to Krishna's memory!")
```

---

## 🚀 Future Deployment

### Hugging Face Spaces
1. Create a Space with Docker runtime
2. Set environment variable: `OLLAMA_HOST=<your-ollama-endpoint>`
3. Push code to the Space repository
4. Requires external Ollama endpoint (GPU Space recommended)

### Railway.app
1. Connect GitHub repo to Railway
2. Add a `Dockerfile` with `chainlit run app.py --host 0.0.0.0`
3. Set `OLLAMA_HOST` to your Ollama server URL
4. Deploy!

> **Note:** For cloud deployment, you'll need a separate Ollama server with GPU, or swap `ChatOllama` for `ChatAnthropic`/`ChatOpenAI`.

---

## 🙏 Acknowledgements

- **Bhagavad Gita Dataset** by [JDhruv14 on HuggingFace](https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita_Dataset)
- **Ollama** for making local LLM inference beautiful
- **Chainlit** for the gorgeous streaming UI
- **LangChain** for the RAG + memory architecture
- **Lord Krishna** — the original author of the Bhagavad Gita 🙏

---

*Hare Krishna Hare Krishna, Krishna Krishna Hare Hare*
*Hare Rama Hare Rama, Rama Rama Hare Hare* 🕉️
