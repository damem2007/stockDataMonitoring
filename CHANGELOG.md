# Changelog

## 2026-08-20

### Backend

#### Added
- Added Yahoo-backed ticker search at `/api/tickers/search`, returning validated ticker suggestions with market, display label, currency, and company name.
- Added yfinance company-name search candidates to `/api/tickers/search`, so users can search by symbol or company name.
- Added TSX shorthand support during onboarding, so selecting TSX and entering `CCO` resolves to `CCO.TO`.
- Added backend ticker validation before persistence so non-existent symbols are rejected instead of saved.
- Added portfolio-level execution plans for trading holdings.
- Added portfolio metric currency context.
- Added `backend/run_api.sh` to start the FastAPI backend on port `8020`.

#### Changed
- Changed backend dev port to `8020` to avoid conflicts with the other local backend.
- Changed ticker search market filtering so alternate listings from unrelated exchanges are not returned for selected TSX/NASDAQ/NYSE filters.
- Changed CORS origin handling so required local frontend origins are merged with configured env origins, preventing stale parent-process env values from blocking the active Next.js dev server.
- Changed trading onboarding persistence so book cost is calculated from `average purchase price * shares` instead of entered manually.
- Confirmed this project uses Python only from `/Users/damidahunsi/stockData/backend/.venv`; no root-level Python venv is required.
- Changed workspace storage identity so unauthenticated guest requests no longer resolve to the local root database user.
- Changed the bootstrapped local root account role from `admin` to `superadmin` to match the system actor model.

#### Fixed
- Fixed `backend/.venv` activation metadata so `source .venv/bin/activate` from the backend folder points to `backend/.venv`.
- Fixed `uvicorn app.main:app --reload` startup from inside the backend folder.
- Fixed FastAPI imports so `app.main:app` works when launched from the `backend` directory.
- Fixed left-panel ticker persistence to validate symbols and surface `422` errors for invalid entries.
- Fixed portfolio holdings persistence so entered holdings can appear in the portfolio dashboard with computed allocation, market value, and return/loss metrics.
- Stopped the stale backend process that was running on port `8000` from the old root-venv command.
- Fixed root README startup instructions so backend commands use `backend/.venv`, `app.main:app`, and port `8020` instead of the removed root venv and old port `8000`.
- Fixed logged-in-vs-guest data leakage where guest workspace requests could display the same database-backed instruments as the logged-in root user.
