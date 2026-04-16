# Architecture Diagrams

These Mermaid diagrams reflect the current codebase structure in `SOP_Chatbot2`.

Related source files:

- `backend/main.py`
- `backend/api/chat.py`
- `backend/api/admin.py`
- `backend/core/rag_chain.py`
- `backend/core/sync.py`
- `backend/rag/loader.py`
- `frontend/src/hooks/useChat.ts`

Standalone Mermaid sources are also saved in this folder as:

- `architecture-flow.mmd`
- `architecture-readme.mmd`
- `chat-sequence.mmd`
- `c4-context.mmd`
- `c4-container.mmd`
- `c4-component.mmd`

## 1. Full Architecture Flow

```mermaid
flowchart TD
    User["User Browser"]

    subgraph FE["Frontend (React + Vite + Radix UI)"]
        App["App Shell"]
        ChatUI["Chat Workspace"]
        AdminUI["Admin Console"]
        Hook["useChat hook"]
    end

    subgraph BE["Backend (FastAPI)"]
        Main["main.py"]
        ChatAPI["/api/chat"]
        CompareAPI["/api/compare"]
        ConvAPI["/api/conversations*"]
        FeedbackAPI["/api/feedback"]
        StatusAPI["/api/sops /api/status /api/health"]
        AdminAPI["/api/admin/*"]
    end

    subgraph CORE["Core Services"]
        Router["Provider selection + fallback"]
        RAG["RAGChain"]
        Rewrite["Conversation-aware query rewrite"]
        Clarify["Clarification path"]
        Retriever["Retriever"]
        Answer["Grounded answer generation"]
        Format["Source formatting + confidence + suggestions"]
        Auth["JWT auth"]
        Sync["SOPSync"]
        Rebuild["Full / incremental reindex"]
        Ingest["Loader + splitter + embeddings"]
    end

    subgraph DATA["Local Data"]
        Chroma["Chroma vector DB"]
        SOPDocs["sop_documents/ PDFs"]
        ImgTxt["img_txt/ extracted text"]
        Flowcharts["flowcharts/ images"]
        Meta["sop_metadata.json"]
        Runtime["data/ conversations, feedback, analytics"]
    end

    subgraph EXT["External Systems"]
        Gemini["Gemini API"]
        Groq["Groq API"]
        SOPSite["Remote SOP site / TOC / PDF links"]
    end

    User --> App
    App --> ChatUI
    App --> AdminUI
    ChatUI --> Hook

    Main --> ChatAPI
    Main --> CompareAPI
    Main --> ConvAPI
    Main --> FeedbackAPI
    Main --> StatusAPI
    Main --> AdminAPI

    Hook -->|REST + SSE| ChatAPI
    Hook --> CompareAPI
    Hook --> ConvAPI
    Hook --> FeedbackAPI
    Hook --> StatusAPI
    AdminUI --> AdminAPI

    ChatAPI --> Router
    CompareAPI --> Router
    Router --> RAG

    RAG --> Rewrite
    RAG --> Clarify
    Rewrite --> Retriever
    Clarify --> Format
    Retriever --> Chroma
    RAG --> Answer
    Answer --> Gemini
    Answer --> Groq
    Answer --> Format

    ConvAPI --> Runtime
    FeedbackAPI --> Runtime
    StatusAPI --> Runtime

    AdminAPI --> Auth
    AdminAPI --> Sync
    AdminAPI --> Rebuild
    AdminAPI --> Runtime

    Sync --> SOPSite
    Sync --> SOPDocs
    Sync --> Ingest
    Rebuild --> Ingest

    SOPDocs --> Ingest
    ImgTxt --> Ingest
    Flowcharts --> Ingest
    Meta --> Ingest
    Ingest --> Chroma
```

## 2. Simplified README Diagram

```mermaid
flowchart LR
    Browser["User Browser"]
    UI["React + Vite UI"]
    API["FastAPI API"]
    RAG["RAG engine"]
    Vector["Chroma vector store"]
    LLM["Gemini / Groq"]
    Files["Local SOP files and metadata"]
    Sync["Admin sync / rebuild"]
    Remote["Remote SOP site"]

    Browser --> UI
    UI --> API
    API --> RAG
    RAG --> Vector
    RAG --> LLM
    Files --> Vector
    Sync --> Remote
    Sync --> Files
    Sync --> Vector
```

## 3. Chat Request Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React UI
    participant API as FastAPI /api/chat
    participant RAG as RAGChain
    participant DB as Chroma
    participant LLM as Gemini or Groq

    User->>UI: Submit question
    UI->>API: POST /api/chat (message, history, provider, answer_mode, source_locked)

    alt Primary provider fails
        API->>API: Mark provider unhealthy
        API->>RAG: Retry with fallback provider
    else Primary provider healthy
        API->>RAG: stream_query(...)
    end

    opt Conversation needs rewrite
        RAG->>LLM: Rewrite follow-up into standalone query
        LLM-->>RAG: Rewritten query
    end

    RAG->>DB: Retrieve relevant chunks
    DB-->>RAG: SOP passages + metadata

    alt Query is ambiguous
        RAG-->>API: Clarification response
        API-->>UI: SSE final event
        UI-->>User: Show clarification options
    else Grounded answer path
        RAG->>LLM: Generate answer from SOP context only
        LLM-->>RAG: Answer draft
        RAG->>RAG: Format sources, confidence, suggestions
        RAG-->>API: Token events + final payload
        API-->>UI: SSE stream
        UI-->>User: Render answer and citations
    end
```

## 4. C4 Context

```mermaid
flowchart LR
    User["Person: Employee or operator"]
    Admin["Person: Admin"]
    System["System: SOP Chatbot 2"]
    LLM["External system: Gemini / Groq"]
    SOPSite["External system: SOP website"]
    Files["Data source: Local SOP files, flowcharts, metadata"]

    User -->|Ask SOP questions| System
    Admin -->|Sync, rebuild, review analytics| System
    System -->|Grounded prompts| LLM
    System -->|Discover and download PDFs| SOPSite
    Files -->|Indexed content| System
```

## 5. C4 Container

```mermaid
flowchart LR
    User["User Browser"]

    subgraph System["SOP Chatbot 2"]
        UI["Container: React SPA<br/>Chat UI, compare UI, admin UI"]
        API["Container: FastAPI app<br/>Chat, admin, health, feedback endpoints"]
        RAG["Container: RAG services<br/>RAGChain, retrieval, provider routing"]
        Store["Container: Chroma DB<br/>Embeddings and document chunks"]
        FileStore["Container: Local file store<br/>PDFs, image text, flowcharts, metadata, logs"]
    end

    LLM["External: Gemini / Groq"]
    SOPSite["External: SOP website"]

    User --> UI
    UI -->|HTTP + SSE| API
    API --> RAG
    RAG --> Store
    RAG --> LLM
    API --> FileStore
    API -->|Sync and PDF discovery| SOPSite
    FileStore -->|Ingestion input| RAG
```

## 6. C4 Component

```mermaid
flowchart TB
    subgraph Backend["FastAPI backend"]
        ChatRouter["Component: Chat router"]
        AdminRouter["Component: Admin router"]
        Auth["Component: JWT auth"]
        Provider["Component: LLM provider manager"]
        RagChain["Component: RAGChain"]
        Retriever["Component: Retriever + source inference"]
        Metadata["Component: Source formatting"]
        Ingestion["Component: Loader + splitter + embeddings"]
        Sync["Component: SOPSync"]
        Feedback["Component: Feedback / conversation store"]
        Chroma["Component: Chroma vector store"]
    end

    UI["React frontend"]
    LLM["Gemini / Groq"]
    SOPSite["Remote SOP website"]
    Files["Local SOP files and runtime data"]

    UI --> ChatRouter
    UI --> AdminRouter

    ChatRouter --> Provider
    ChatRouter --> RagChain
    ChatRouter --> Feedback

    Provider --> LLM
    RagChain --> Provider
    RagChain --> Retriever
    RagChain --> Metadata
    Retriever --> Chroma

    AdminRouter --> Auth
    AdminRouter --> Sync
    AdminRouter --> Ingestion
    AdminRouter --> Feedback

    Sync --> SOPSite
    Sync --> Files
    Ingestion --> Files
    Ingestion --> Chroma
```
