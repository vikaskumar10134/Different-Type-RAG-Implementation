# Self-RAG using LangGraph + NVIDIA AI + FAISS

A Self-Reflective Retrieval-Augmented Generation (Self-RAG) implementation built with **LangGraph**, **LangChain**, **NVIDIA LLM**, **NVIDIA Embeddings**, and **FAISS**.

The workflow intelligently decides whether retrieval is necessary, verifies retrieved documents, validates generated answers, and automatically retries with an improved retrieval query when needed.

---

# Features

* Intelligent retrieval decision
* FAISS vector database
* NVIDIA Embeddings
* Document relevance filtering
* Context-only answer generation
* Answer grounding verification
* Query rewriting
* Automatic retry mechanism
* LangGraph workflow

---

# Tech Stack

| Component         | Library                        |
| ----------------- | ------------------------------ |
| LLM               | NVIDIA Chat                    |
| Workflow          | LangGraph                      |
| Embeddings        | NVIDIA Embeddings              |
| Vector Store      | FAISS                          |
| PDF Loader        | PyPDFLoader                    |
| Chunking          | RecursiveCharacterTextSplitter |
| Structured Output | Pydantic                       |

---

# Workflow Overview

```mermaid
flowchart TD

A[User Question]

A --> B{Need Retrieval?}

B -->|No| C[Generate Direct Answer]

B -->|Yes| D[Retrieve Documents]

C --> Z[END]

D --> E[Check Document Relevance]

E -->|Relevant| F[Generate From Context]

E -->|No Relevant Documents| Y[No Answer Found]

F --> G[Verify Answer Support]

G -->|Supported| H[Check Answer Usefulness]

G -->|Not Supported| I[Revise Answer]

I --> G

H -->|Useful| Z

H -->|Not Useful| J[Rewrite Query]

J --> D

Y --> Z
```

---

# Step 1 - Document Preprocessing

Before answering any question, the documents are prepared for semantic search.

## Pipeline

```text
PDF Files
     │
     ▼
Load Documents
     │
     ▼
Split into Chunks
     │
     ▼
Generate Embeddings
     │
     ▼
Store in FAISS
     │
     ▼
Retriever
```

### Process

1. Load multiple PDF documents.
2. Split documents into small overlapping chunks.
3. Generate embeddings for every chunk.
4. Store embeddings inside FAISS.
5. Create a retriever using MMR search.

---

# Step 2 - Decide Whether Retrieval is Needed

Instead of always searching the vector database, the model first decides whether retrieval is required.

Examples

| Question                            | Retrieval |
| ----------------------------------- | --------- |
| What is Python?                     | ❌ No      |
| What is our company's leave policy? | ✅ Yes     |

---

```mermaid
flowchart LR

Q[Question]

Q --> D{Need Retrieval?}

D -->|Yes| R[Retrieve]

D -->|No| G[Generate Direct Answer]
```

---

# Step 3 - Direct Generation

If retrieval is unnecessary, the LLM answers using its general knowledge.

```text
Question
    │
    ▼
LLM
    │
    ▼
Answer
```

---

# Step 4 - Retrieve Documents

The retriever searches the FAISS vector database.

```text
Question
     │
     ▼
Retriever
     │
     ▼
Top 3 Similar Documents
```

---

# Step 5 - Document Relevance Check

Each retrieved document is checked independently.

Only relevant documents continue to the next stage.

```mermaid
flowchart TD

R[Retrieved Documents]

R --> D1[Document 1]

R --> D2[Document 2]

R --> D3[Document 3]

D1 --> K1[Relevant]

D2 --> K2[Not Relevant]

D3 --> K3[Relevant]
```

After filtering:

```text
Relevant Documents
        │
        ▼
Merged Context
```

---

# Step 6 - Generate Answer From Context

The filtered documents are combined into one context.

```text
Relevant Documents
        │
        ▼
Context
        │
        ▼
LLM
        │
        ▼
Answer
```

The prompt instructs the model to:

* Use only the provided context
* Never use outside knowledge
* Return "No relevant document found" if context is insufficient

---

# Step 7 - Verify Answer Support

The generated answer is compared against the retrieved context.

Possible outputs:

* Fully Supported
* Partially Supported
* No Support

```mermaid
flowchart LR

A[Generated Answer]

C[Retrieved Context]

A --> V[Support Verification]

C --> V

V --> F[Fully Supported]

V --> P[Partially Supported]

V --> N[No Support]
```

---

# Step 8 - Revise Unsupported Answers

If the answer is not sufficiently supported, it is regenerated.

```mermaid
flowchart LR

A[Answer]

A --> R[Revise]

R --> V[Verify Again]

V -->|Still Unsupported| R

V -->|Supported| U[Continue]
```

This loop continues until:

* Answer is supported
* Maximum retry count is reached

---

# Step 9 - Check Answer Usefulness

A grounded answer is not always useful.

Example

Question

> What is the refund policy?

Poor Answer

> Our company values customers.

Although it may be true, it does not answer the question.

```mermaid
flowchart TD

A[Generated Answer]

A --> U{Useful?}

U -->|Yes| E[END]

U -->|No| Q[Rewrite Query]
```

---

# Step 10 - Rewrite Query

If the answer is not useful, the original question is rewritten into a better retrieval query.

Example

Original

```text
Can I get my money back?
```

Rewritten

```text
refund policy cancellation refund timeline
```

The improved query is sent back to the retriever.

```mermaid
flowchart LR

Q[Original Question]

Q --> W[Rewrite Query]

W --> R[Retrieve Again]
```

---

# Complete LangGraph Pipeline

```mermaid
flowchart TD

Start([START])

Start --> Decide[Decide Retrieval]

Decide -->|No| Direct[Generate Direct]

Direct --> End([END])

Decide -->|Yes| Retrieve[Retrieve Documents]

Retrieve --> Relevant[Check Relevance]

Relevant -->|Relevant| Generate[Generate From Context]

Relevant -->|No Relevant Docs| NoAnswer[No Answer]

Generate --> Verify[Verify Support]

Verify -->|Supported| Useful[Check Usefulness]

Verify -->|Not Supported| Revise[Revise Answer]

Revise --> Verify

Useful -->|Useful| End

Useful -->|Not Useful| Rewrite[Rewrite Query]

Rewrite --> Retrieve

NoAnswer --> End
```

---

# Retry Strategy

## Answer Revision Loop

```text
Generate Answer
      │
      ▼
Verify Support
      │
      ▼
Revise Answer
      │
      └───────────────► Verify Again
```

---

## Query Rewrite Loop

```text
Question
    │
    ▼
Retrieve
    │
    ▼
Generate
    │
    ▼
Useful?
    │
    ▼
Rewrite Query
    │
    └────────────► Retrieve Again
```

---

# Project Structure

```text
Self_RAG_Module.py

│
├── Document Preprocessing
├── State Definition
├── Retrieval Decision
├── Direct Generation
├── Retrieve Documents
├── Relevance Checking
├── Context Generation
├── Support Verification
├── Answer Revision
├── Useful Check
├── Query Rewriting
└── LangGraph Workflow
```

---

# Advantages

* Reduces unnecessary retrieval
* Improves retrieval quality
* Filters irrelevant documents
* Minimizes hallucinations
* Verifies answer grounding
* Automatically retries when answers are poor
* Modular LangGraph architecture

---

# Future Improvements

* Hybrid Search (BM25 + Dense Retrieval)
* Cross Encoder Re-ranking
* Parent Document Retriever
* Multi Query Retrieval
* Context Compression
* Streaming Responses
* Human-in-the-Loop
* Memory Integration
* Citation Generation
* Web Search Fallback

---

# Overall Lifecycle

```text
User Question
      │
      ▼
Need Retrieval?
      │
 ┌────┴────┐
 │         │
 ▼         ▼
Direct   Retrieve
           │
           ▼
Relevance Check
           │
           ▼
Generate Answer
           │
           ▼
Support Verification
           │
           ▼
Useful Check
           │
      ┌────┴────┐
      │         │
      ▼         ▼
     END   Rewrite Query
                │
                ▼
           Retrieve Again
```

---

**This implementation demonstrates a complete Self-RAG pipeline where the system not only retrieves and generates answers, but also evaluates its own outputs, revises unsupported responses, and improves retrieval through query rewriting for higher reliability.**
