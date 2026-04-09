# SOP Chatbot 2

Windows-first full-stack SOP chatbot with:

- FastAPI backend
- React + Vite frontend
- Chroma vector store
- Gemini and Groq LLM support
- Admin tools for sync, rebuild, analytics, and feedback review

This guide covers fresh setup, configuration, running locally, rebuilding the index, and troubleshooting.

## 1. Project Overview

The app loads SOP content from local files, chunks and embeds it into Chroma, and exposes a chat API that the React frontend consumes.

Main content sources used for indexing:

- `sop_documents/` for PDF SOPs
- `img_txt/` for extracted text files
- `flowcharts/` for image-based workflow references
- `sop_metadata.json` for SOP links, versions, and metadata

At startup, the backend:

1. Tries to load an existing Chroma database from `chroma_db/`
2. Builds a new one if no database exists
3. Initializes the default LLM provider
4. Serves the API on port `8000`

The frontend runs on port `5173` during development and talks to the backend through the Vite proxy.

## 2. Repository Layout

```text
backend/              FastAPI app, RAG logic, admin/chat APIs
frontend/             React + Vite frontend
sop_documents/        PDF SOP source files
img_txt/              Extracted text documents
flowcharts/           Flowchart images exposed as static assets
data/                 Feedback, failed queries, conversation logs
chroma_db/            Persisted Chroma vector database
tests/                Backend-focused tests
start.ps1             Starts backend + frontend together
stop.ps1              Stops managed processes started by start.ps1
rebuild_chroma.py     Full local vectorstore rebuild script
requirements.txt      Backend Python dependencies
```

## 3. Prerequisites

Install these before setup:

- Python `3.10+`
- Node.js `18+` and `npm`
- PowerShell
- Internet access for:
  - installing dependencies
  - first-time model/provider validation
  - first-time embedding/model downloads from Python packages

You also need at least one LLM provider API key:

- Gemini: `gemini_api_key`
- Groq: `groq_api_key`

The configured `llm_provider` must have a valid key, otherwise backend startup can fail.

## 4. Fresh Setup

### 4.1 Clone and enter the project

```powershell
git clone <your-repo-url>
cd SOP_Chatbot2
```

### 4.2 Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

### 4.3 Install backend dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 4.5 Create your environment file

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set at minimum:

- `llm_provider`
- one valid API key for that provider
- `admin_password`
- `jwt_secret`

## 5. Environment Configuration

The backend reads settings from `.env` through `backend/config.py`.

Important values:

- `llm_provider`: `gemini` or `groq`
- `gemini_api_key`
- `gemini_model`
- `groq_api_key`
- `groq_model`
- `admin_password`
- `jwt_secret`

Optional path overrides exist, but the defaults already point at the repo folders:

- `sop_documents_dir`
- `flowcharts_dir`
- `img_txt_dir`
- `chroma_db_dir`
- `sop_metadata_path`
- `data_dir`
- `sop_base_url`

Minimal example:

```env
llm_provider=gemini
gemini_api_key=your_real_key_here
gemini_model=gemini-2.5-pro
groq_api_key=
groq_model=llama-3.1-8b-instant
admin_password=change-this-password
jwt_secret=replace-with-a-long-random-secret
```

## 6. Content and Data Requirements

Before first run, verify these folders/files exist:

- `sop_documents/`
- `img_txt/`
- `flowcharts/`
- `sop_metadata.json`

Notes:

- PDF files in `sop_documents/` are indexed automatically
- `.txt` files in `img_txt/` are indexed automatically
- `.png`, `.jpg`, and `.jpeg` in `flowcharts/` are added as flowchart references
- `data/` stores feedback, failed queries, and conversation history

If `chroma_db/` does not exist, the backend will build it on first startup.

## 7. Running the Project

### 7.1 Recommended: start both services with the manager script

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

This starts:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

The script will:

1. find Python from `venv`, `.venv`, or PATH
2. find `npm`
3. start backend and frontend
4. store process IDs in `.run/processes.json`
5. write logs under `.run/`

### 7.2 Stop managed services

```powershell
powershell -ExecutionPolicy Bypass -File .\stop.ps1
```

### 7.3 Run services manually

Backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend in a second terminal:

```powershell
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## 8. First Startup Expectations

The first backend startup can take noticeably longer because it may:

1. read all PDFs and text files
2. generate chunks
3. create embeddings
4. build the Chroma database in `chroma_db/`
5. download model assets needed by embedding dependencies

If the backend is still starting, wait until `http://127.0.0.1:8000/api/health` responds successfully.

## 9. Verify the Setup

After startup, check these URLs:

- Frontend: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- Backend health: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- FastAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Expected health response includes:

- `status`
- `llm_provider`
- `model`
- `available_models`
- `provider_status`

## 10. Frontend Build

To create a production frontend build:

```powershell
cd frontend
npm run build
```

When `frontend/dist/` exists, the FastAPI app can serve it directly in production mode.

## 11. Rebuilding or Refreshing the Vector Database

Use this when:

- SOP files were added, removed, or updated
- extracted text files changed
- metadata changed significantly
- the Chroma DB is stale or corrupted

### 11.1 Rebuild with the helper script

Stop the backend first, then run:

```powershell
.\venv\Scripts\python.exe .\rebuild_chroma.py
```

This will:

1. reload SOP documents and text files
2. split documents into chunks
3. delete and recreate `chroma_db/`
4. print chunk/source summary information

### 11.2 Rebuild from the admin UI

After logging into the admin panel in the frontend, you can trigger:

- sync
- rebuild index

Important:

- rebuilding can fail if another backend process still has `chroma_db/` open
- if that happens, stop running backend processes and retry

## 12. Logs and Runtime Files

Managed runs write to `.run/`.

Important files:

- `.run/processes.json` stores tracked PIDs
- `.run/manager.log` stores lifecycle events
- `.run/backend/<timestamp>.out.log`
- `.run/backend/<timestamp>.err.log`
- `.run/frontend/<timestamp>.out.log`
- `.run/frontend/<timestamp>.err.log`

If startup fails, check the latest backend and frontend logs first.

## 13. Admin Access

The frontend admin dialog uses the backend admin login endpoint.

Set these in `.env`:

- `admin_password`
- `jwt_secret`

Default values exist in code, but you should override them for any real use.

Admin features include:

- sync SOP sources
- rebuild index
- view analytics
- review feedback
- review failed queries

## 14. Running Tests

Activate the venv first, then run:

```powershell
pytest
```

If `pytest` is not available in your environment, install it explicitly:

```powershell
pip install pytest
```

You can also run a specific test file:

```powershell
pytest tests\test_chat_api.py
```

## 15. Common Troubleshooting

### PowerShell blocks scripts

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

### Backend does not start

Check:

- `.env` exists
- `llm_provider` is valid
- the selected provider has a real API key
- `venv` dependencies installed successfully
- latest backend log in `.run/backend/`

### Frontend cannot connect to backend

Check:

- backend is running on `8000`
- frontend is running on `5173`
- `http://127.0.0.1:8000/api/health` responds
- Vite dev server was started from `frontend/`

### Chroma rebuild fails with a lock or permission error

Stop all running backend or rebuild processes first:

```powershell
.\stop.ps1
```

Then retry:

```powershell
.\venv\Scripts\python.exe .\rebuild_chroma.py
```

### First run is very slow

This is normal if:

- embeddings are being generated for the first time
- the vectorstore is being built for the first time
- local model assets are downloading

### No answers or poor retrieval quality

Check:

- SOP files are actually present in `sop_documents/`
- text extractions are present in `img_txt/`
- `sop_metadata.json` is valid JSON
- rebuild the vectorstore after changing source documents

## 16. Quick Start Summary

For a standard local setup on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
cd frontend
npm install
cd ..
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Then open:

- `http://127.0.0.1:5173`

