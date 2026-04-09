# SOP Chatbot 2

Windows-first setup and run instructions for the backend and frontend.

## Prerequisites

- Python 3.10+
- Node.js and npm
- PowerShell

## Initial Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Start Both Services

This starts:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Logs are written to `.run\`.

## Stop Both Services

```powershell
powershell -ExecutionPolicy Bypass -File .\stop.ps1
```

## Run Services Manually

Start backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Start frontend in a separate terminal:

```powershell
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## Notes

- `start.ps1` stores process IDs in `.run\processes.json`.
- `stop.ps1` uses that file to stop both services cleanly.
- Each run creates a separate log session:
  - Backend stdout: `.run\backend\<timestamp>.out.log`
  - Backend stderr: `.run\backend\<timestamp>.err.log`
  - Frontend stdout: `.run\frontend\<timestamp>.out.log`
  - Frontend stderr: `.run\frontend\<timestamp>.err.log`
- Lifecycle events are appended to `.run\manager.log`.
- If chat or admin login shows backend-related errors, confirm `http://127.0.0.1:8000/api/health` is reachable after running `.\start.ps1`.
- If PowerShell blocks script execution, keep using the `-ExecutionPolicy Bypass` form shown above.
