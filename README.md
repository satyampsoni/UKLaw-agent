# ⚖️ UK LawAssistant — Sovereign AI Legal Intelligence

A **Retrieval-Augmented Generation (RAG)** system that answers questions about UK legislation with pinpoint accuracy. Ask a question in plain English, get an answer grounded in the actual statutory text — with section-level citations you can verify.

Built on the [Relax AI API](https://www.civo.com/relax-ai) by **CIVO**, it runs the **DeepSeek-V31-Terminus** model through a sovereign, EU/UK-hosted inference endpoint — no data leaves compliant infrastructure.

### What's in the database

| Act | Sections | Coverage |
|-----|----------|----------|
| **Data Protection Act 2018** | 3,347 nodes | GDPR implementation, data rights, enforcement |
| **Online Safety Act 2023** | 3,831 nodes | Platform duties, illegal content, age verification |
| **Consumer Rights Act 2015** | 1,163 nodes | Faulty goods, unfair terms, digital content |

> **8,341 legal nodes** totalling **825,950 characters** of statutory text, indexed with FTS5 full-text search.

---

## How It Works and different from ChatGPT

When you ask ChatGPT *"What are the lawful bases for processing personal data?"*, it answers from training data — which may be outdated, incomplete, or hallucinated. It will confidently cite *"Section 12(3) of the Data Protection Act"* even if that section doesn't exist.

**UK LawAssistant takes a fundamentally different approach:**

```
Your Question
     │
     ▼
┌─────────────┐    FTS5 + BM25     ┌──────────────────┐
│  Search     │───────────────────▶│  SQLite Database  │
│  Engine     │◀───────────────────│  8,341 legal nodes│
└─────────────┘    Ranked sections └──────────────────┘
     │
     │  Top 5 sections (real statutory text)
     ▼
┌─────────────┐    System prompt:  ┌──────────────────┐
│  RAG Prompt │   "cite sections,  │  Relax AI API    │
│  Builder    │───don't invent"───▶│  (CIVO)          │
└─────────────┘                    │  DeepSeek-V31    │
     │                             └──────────────────┘
     │  Grounded answer with citations
     ▼
┌─────────────┐
│  Your       │
│  Answer     │  "Section 6(1) of the DPA 2018 provides..."
└─────────────┘
```

The LLM never generates from memory. It only **explains** legislation that the search engine has already retrieved and placed in the prompt. Every claim is traceable to a real section.

### Relax AI API by CIVO

The LLM inference runs through [CIVO's Relax AI](https://www.civo.com/relax-ai) — a sovereign AI platform that provides OpenAI-compatible API endpoints hosted within EU/UK data centres. This matters for legal applications because:

- **Data sovereignty** — queries about sensitive legal matters never leave compliant infrastructure
- **Model**: `DeepSeek-V31-Terminus` — a high-capability reasoning model
- **Temperature**: `0.1` — near-deterministic for legal precision (the same question gives the same answer)
- **Max tokens**: `5,000` — enough for detailed explanations with full citations

### Frontend Parameters

The UI exposes three controls that shape how the system responds:

| Parameter | Options | Effect |
|-----------|---------|--------|
| **Scope** | All Acts / DPA 2018 / OSA 2023 / CRA 2015 | Restricts FTS5 search to one Act — eliminates cross-Act noise |
| **Strict Mode** | On / Off | **Off** (default): LLM explains the law in plain English using the context. **On**: LLM only quotes the statutory text verbatim — no interpretation |
| **Search** (bottom panel) | Free text | Searches legislation directly without calling the LLM — browse raw sections |


## Local Setup

### Prerequisites

- **Python 3.12+** (tested on 3.14)
- A **Relax AI API key** from [CIVO](https://www.civo.com/relax-ai)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/uklaw-assistant.git
cd uklaw-assistant
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
RELAX_AI_API_KEY="your-relax-ai-api-key-here"
RELAX_AI_BASE_URL=https://api.relax.ai/v1
RELAX_AI_MODEL="DeepSeek-V31-Terminus"
RELAX_AI_MAX_TOKENS=5000
RELAX_AI_TEMPERATURE=0.1
```

### 4. Seed the database

This fetches 3 UK Acts from legislation.gov.uk and builds the search index:

```bash
python -m scripts.seed_database
```

Expected output: `8,341 nodes` across 3 Acts (~825K characters).

### 5. Start the server

```bash
uvicorn app.api.main:app --reload --port 8000
```

Open **http://localhost:8000** — the frontend loads automatically.

API docs available at **http://localhost:8000/docs** (Swagger UI).

### Project Structure

```
uklaw-assistant/
├── app/
│   ├── api/          # FastAPI routes, schemas, app factory
│   ├── config.py     # Typed config from .env (pydantic-settings)
│   ├── ingestion/    # Fetch XML → parse → seed database
│   ├── llm/          # Relax AI async client (chat, stream, retry)
│   ├── models/       # SQLite + FTS5 database, LegalNode data model
│   ├── rag/          # RAG pipeline + prompt templates
│   └── search/       # Search engine (FTS5 → section expansion → dedup)
├── static/           # Frontend (HTML, CSS, JS — no build step)
├── data/             # SQLite database (auto-generated by seed)
├── scripts/          # CLI tools: seed, test_llm, test_search, test_rag
└── requirements.txt
```

---

## Questions to Try

These are good starting questions that exercise different parts of the system:

### Data Protection Act 2018

> **What are the lawful bases for processing personal data under UK law?**
>
> *Tests: broad search across Part 2, multiple section retrieval*

> **What is a data protection officer and what are their responsibilities?**
>
> *Tests: sections 69–71, cross-section context assembly*

> **When can personal data be erased under the DPA 2018?**
>
> *Tests: section 47 — right to erasure with conditions and exceptions*

### Online Safety Act 2023

> **What duties do platforms have regarding illegal content?**
>
> *Tests: Part 3, Chapter 2–3, multi-section response*

> **What are the age verification requirements under the Online Safety Act?**
>
> *Tests: children's safety duties, detailed cross-referencing*

### Consumer Rights Act 2015

> **What are my rights if I buy faulty goods?**
>
> *Tests: Part 1, consumer-friendly language generation*

> **What makes a contract term unfair under UK consumer law?**
>
> *Tests: Part 2, sections 62–70, schedule references*

### Cross-Act Questions (Scope: All Acts)

> **How does UK law protect children online?**
>
> *Tests: OSA 2023 children's duties + DPA 2018 data protection for minors*

> **What are the penalties for data breaches in the UK?**
>
> *Tests: DPA enforcement provisions, cross-Part retrieval*

### Pro Tips

- Use the **Scope** dropdown to restrict to one Act for more focused answers
- Toggle **Strict Mode** to get pure statutory quotes without interpretation
- Use the **Search** bar at the bottom to browse legislation directly without asking the LLM
- Click on **source cards** in the right panel to expand the full statutory text that was consulted

---

##  Future Scopes

This MVP proves the core loop: **ingest → search → ground → answer**. But the architecture is designed to scale into a production system that UK government departments, courts, and private law firms could use with Relax AI as the sovereign inference backbone.

### Scaling the Knowledge Base

| What | How | Impact |
|------|-----|--------|
| **All UK primary legislation** | legislation.gov.uk has 3,000+ Acts — the ingestion pipeline already handles arbitrary Acts, just expand `MVP_ACTS` | Full statutory coverage |
| **Statutory Instruments (SIs)** | Secondary legislation (regulations, orders) uses the same XML schema — minimal parser changes | Covers the rules that implement Acts |
| **Case law integration** | Ingest from [National Archives](https://caselaw.nationalarchives.gov.uk/) / BAILII — add a `case_law` node type with judge, court, date fields | "What did the court decide in *Lloyd v Google*?" |
| **Amendment tracking** | legislation.gov.uk provides amendment metadata — track which sections are amended, repealed, or prospective | "Is section 47 still in force?" with confidence |
| **Devolved legislation** | Scottish Parliament (asp), Welsh Senedd (asc), Northern Ireland Assembly (nia) — same XML API, different `act_type` | UK-wide coverage |

### Government & Institutional Use Cases

**Parliamentary Drafting Office**
> A bill drafter asks: *"Show me every existing provision that references 'age verification' across all Acts."*
> The system returns a cross-statute audit in seconds — work that currently takes days of manual searching across Westlaw and Hansard.

**Ministry of Justice — Policy Impact Assessment**
> *"If we amend section 64 of the DPA 2018, which other sections reference it?"*
> Cross-reference analysis powered by the node tree. Add amendment simulation: show what the Act looks like with the proposed change applied.

**Courts & Tribunals**
> Judges preparing for hearings could query: *"What are the statutory defences available under Part 3 of the Online Safety Act?"*
> With case law integration, the answer includes both the statute and how courts have interpreted it.

**Citizens Advice & Legal Aid**
> Non-lawyers get plain-English explanations scoped to their situation: *"I bought a faulty washing machine — what are my rights?"*
> Strict mode off gives accessible language; strict mode on gives the exact statutory text a solicitor needs.

### Technical Extensions with Relax AI

| Extension | Description |
|-----------|-------------|
| **Multi-model routing** | Use `Mistral-7b-embedding` (available on Relax AI) for semantic vector search alongside FTS5 — hybrid retrieval for both keyword precision and conceptual similarity |
| **Streaming answers** | The pipeline already supports `ask_stream()` — wire it to Server-Sent Events for real-time typing in the frontend |
| **Batch analysis** | Process a set of contract clauses against the CRA 2015 unfair terms provisions — bulk compliance checking for law firms |
| **Multi-turn legal research sessions** | The conversation history already tracks 3 turns — extend with persistent sessions so a researcher can build a complex argument across multiple queries |
| **Confidence scoring** | Use the BM25 scores + section coverage to estimate how well the retrieved context covers the question — warn the user when coverage is low |
| **Audit trail** | Log every query, the sections retrieved, the prompt sent, and the answer generated — full transparency for regulated environments |
| **Fine-tuned legal summarisation** | Use Relax AI's model catalogue to test different models (Kimi-K25, Llama-4-Maverick) for legal summarisation quality — no infrastructure changes, just swap the model name in `.env` |

### Why Relax AI Makes This Viable for Government

Traditional AI deployments in government face three blockers: **data sovereignty** (can't send legal queries to US-hosted APIs), **vendor lock-in** (tied to one model provider), and **cost unpredictability** (token-based billing at scale).

Relax AI by CIVO addresses all three:

- **Sovereign infrastructure** — hosted in UK/EU data centres, compliant with UK data protection requirements by default
- **Model flexibility** — OpenAI-compatible API means switching from DeepSeek to Llama to Mistral is a one-line config change, no code rewrite
- **Predictable scaling** — CIVO's cloud-native platform means the inference layer scales with demand without per-token surprises

This positions UK LawAssistant not as a chatbot, but as **legal infrastructure** — a sovereign, auditable, citation-grounded system that could sit alongside existing tools like Westlaw and LexisNexis, but powered by open models on compliant infrastructure.
