# Changelog

## 2026-09-09
- Restricted the Advanced Trading indicator endpoint to authenticated users so it remains a Portfolio-only capability.
- Added an Advanced Trading indicator service that scores BUY/SELL timing candidates from the existing OHLCV indicator engine and lets OpenAI rank only verified historical dates and prices, with deterministic local interpretation as the fallback.
- Scoped prediction runs to active authenticated portfolio instruments with a lightweight server-side membership check, including direct standalone prediction URLs.
- Optimized direct portfolio membership verification to a single user/symbol/active SQL existence query, avoiding the general instrument-store load during URL validation.
- Added session-scoped portfolio instrument membership validation before direct portfolio symbol analysis requests, returning a clear missing-symbol result instead of allowing an endless analysis load.
- Added a separate authenticated quantitative prediction module at /api/predictions: it consumes the existing instrument analysis as an explicit input and produces independent 1-, 5-, 10-, 20-, and 60-day direction/return forecasts for one portfolio instrument at a time.
- Added prediction features for multi-year trading-centre/reversion distance, pullback-from-high, range position, volatility, volume z-score, and candle range/body context, with leakage-controlled sequential folds, regime-aware logistic/gradient-boosting weighting, baseline comparisons, precision/recall/F1/MCC/ROC-AUC/Brier/calibration metrics, feature-impact explanations, monitoring drift flags, and a cost-aware walk-forward backtest.
- Added structured OpenAI prediction interpretation with compact evidence prompts. OpenAI is interpretation-only; local quantitative evidence remains authoritative and deterministic interpretation is the final fallback.
- Added prediction_provider_diagnostics persistence and a diagnostics endpoint so OpenAI timeouts, HTTP failures, parse failures, and unusable structured responses retain sanitized provider metadata and a bounded response excerpt for assessment.
- Added authenticated portfolio prediction picker/menu access and a Prediction tab on portfolio instrument pages.
- Updated the Market Explorer loader to keep the honest five-stage progress contract while rotating the requested final-synthesis sub-stages during long-running stage 5 / 5 work.
- Restored execution-checklist visibility for provider-proposed symbols, including rows outside the final selected-instrument list; the checklist no longer filters those rows out.
- Changed each explicit Market Explorer run to refresh Yahoo-backed candidate evidence before ranking, while retaining cached data as the source fallback when a live fetch fails. Identical refreshed evidence can still produce the same deterministic ranking.

### Backend

#### Changed
- Added a central `app/config.py` settings module for environment-backed AI disclosures, OpenAI endpoint/model/timeouts/token limits, provider notes, and result schema version, with safe non-secret defaults.
- Clarified that local analysis is the reproducible OHLCV/evidence and validation layer, not the preferred AI analyst; OpenAI remains the interpretation provider and local rules remain the final fallback.
- Clarified the analysis-provider contract: deterministic local calculations run first as the reproducible Yahoo-backed evidence and validation layer that OpenAI interprets; local rules are only the final fallback when OpenAI cannot return an interpretation.
- Moved Market Explorer and instrument-tab disclosure, OpenAI endpoint/model, timeout, token, reasoning, future-provider, and result-schema settings behind environment variables with type-safe defaults.
- Changed Market Explorer budget distribution to enforce the user's selected allocation policy and prevent AI-generated synthetic cash rows from bypassing authoritative amounts and currencies.
- Changed the Market Explorer OpenAI request to use the dedicated `OPENAI_MARKET_EXPLORER_MODEL` setting, retaining `gpt-5.6-sol` as the quality-focused default rather than switching to the lower-capability BA_Automation model.
- Changed the request to cap generated output at 3,000 tokens, use low bounded reasoning effort for GPT-5 models, and allow a 90-second read window so a valid deep analysis is not prematurely sent to the local fallback.
- Changed Market Explorer provider ordering to attempt OpenAI first, reserve Qwen for its future provider slot, and use deterministic rules-based analysis only as the final fallback.
- Changed Market Explorer budgets to accept one base currency at a time while preserving the separate CAD/USD/Both plan-currency selection; selected markets, not currency, determine the instrument universe.

#### Added
- Added an instrument AI-enhancement endpoint that receives the local Yahoo-backed analysis and asynchronously returns structured interpretations for Signal, Chart, ML, Backtest, News, and Portfolio Fit without replacing raw local metrics.
- Added Yahoo Finance financial-statement acquisition and caching for income statement, balance sheet, and cash-flow rows, with conservative local statement-growth evidence added to signal scoring.
- Added VIX acquisition and macro-context enrichment to local regime scoring, including high-stress confirmation and sizing caution.
- Added complete instrument provenance payloads covering OHLCV, financial statements, earnings, news, benchmark, VIX, feature engineering, local ML validation, and backtest inputs.
- Added OpenAI request telemetry for instrument interpretation, including model, request size, status, elapsed time, and sanitized fallback category.
- Added a per-instrument Market Explorer analysis engine that runs multi-year trading-centre/reversion, pullback-band and entry-zone, earnings/event, rich-news, PESTEL, thesis-classification, and base/worst/best-case scenario modules before recommendation synthesis.
- Added module-level evidence to the OpenAI candidate prompt so detailed news, earnings metrics, historical range statistics, and user risk/strategy/intent criteria are available for every reviewed instrument.
- Added local execution-plan synthesis from the same modules so the final fallback still produces instrument-specific entry guidance, scenario checks, thesis conclusions, and monitoring steps.
- Added five-year daily Market Explorer history requests, with separate recent one-year movement context and multi-year trading-centre/range statistics.
- Added a Market Explorer budget-use question for full deployment, investment percentage, or fixed cash reserve, with base-currency cash reserve amounts returned explicitly.
- Added structured OpenAI provider telemetry for endpoint, model, attempt, response status, elapsed time, and sanitized failure category without logging API keys, prompts, or response bodies.
- Added redacted OpenAI request diagnostics for request size, candidate count, timeout phase, and response timing so provider availability can be distinguished from model-generation latency.
- Added a safe fallback reason to the result disclosure so users can distinguish missing backend configuration, timeout, HTTP failure, request failure, and unusable OpenAI output.
- Added Yahoo Finance `USDCAD=X` conversion metadata for mixed CAD/USD allocations, including quote source, UTC quote time, and a confirmation warning because FX rates change.
- Added a versioned Market Explorer result contract with explicit run, request, summary, evidence, recommendation, allocation, execution, context, and disclosure sections shared by OpenAI and local rules.
- Changed provider reconciliation to retain reviewed candidates and the local ranking baseline, union provider and local selections, and preserve local market facts when a provider returns partial instrument data.
- Added explicit excluded-instrument reporting so preferred workspace symbols are shown with a reason when they cannot be ranked or are not selected for the current plan.
- Added execution-plan reconciliation so an empty or partial AI execution plan is completed from the local evidence plan for every viable selected instrument.

## 2026-08-30

### Backend

#### Changed
- Changed portfolio reads to use cached/hydrated data by default and reserve forced refreshes for explicit sync calls, making initial account hydration faster without showing a misleading empty portfolio first.
- Changed portfolio inference notes to include execution-plan context derived from price, trend, news, and signal analysis so richer portfolio findings appear under the dashboard Inference section.
- Changed refresh interval configuration to accept both `REFRESH_INTERVAL_SECONDS` spelling.
- Changed analysis pricing so Summary, Chart, ML, Backtest, portfolio rows, and alerts share a fresh latest-market overlay instead of mixing stale snapshot prices with period-specific history cache closes.
- Changed Yahoo history timestamps to preserve UTC during backend serialization, showing exchange/source timestamps with timezone context instead of backdating UTC rows to local dates.
- Changed manual workspace sync and portfolio loading to force scoped refreshes for active instruments instead of accepting day-old intraday cache data.
- Changed the application-scope refresh loop to recompute execution plans, refresh symbol-specific news, and evaluate alert triggers on the backend interval without depending on an open frontend session.
- Changed the refresh loop to evaluate the local workspace plus each persisted user workspace so mixed local, guest-warmed, and authenticated instruments remain covered.
- Changed account verification resend so an expired but recognizable verification token can issue a fresh verification link without asking for email or username.
- Changed verification resend responses to keep generic success messaging while returning a clear stale-account error when a token points to an account that no longer exists.
- Changed portfolio and watchlist price-date payloads so all supported currencies and markets expose normalized intraday quote timestamps with UTC context for clearer local/UTC display.
- Added the authenticated `/api/ai-market-explorer/flow` and `/api/ai-market-explorer/analyze` process-flow endpoints, with OpenAI as the active provider and a local-rules response plus future Qwen provider boundary as the fail-safe path.
- Added `Both` to the market-explorer currency question so users can assess CAD and USD opportunities together.
- Added `Not sure` to the market-explorer intent question; evidence analysis uses a neutral Hold/Watch read while preserving the user's uncertainty in the generated profile.
- Added `intentOptions` to market-explorer profile payloads so `Not sure` makes Buy/Add, Sell/Trim, and Hold/Watch available for model comparison and strategy alternatives.
- Added returned ticker currencies to market-explorer allocations and ranked evidence, plus longer-horizon/YTD trend context so peak-only pullbacks are not treated as sufficient growth-entry evidence.
- Added authenticated persistence for Market Explorer analysis history, including save, list, delete, delete-all, and refresh-from-saved-input endpoints.
- Changed market-explorer sector matching to support multiple selected sectors while retaining `Infer` as the no-selection fallback.
- Fixed market-explorer currency normalization so `Both` resolves to the `CAD`/`USD` currency array and never reaches numeric formatting as `BOTH`.

## 2026-08-29

### Backend

#### Added
- Added `/api/auth/verify-account` so registration verification links generated by `create_account_verification_token()` can be redeemed.

#### Fixed
- Fixed account verification table creation during database initialization.
- Fixed ORM import compatibility so auth modules can load under both `backend.app.*` and `app.*` execution styles.

## 2026-08-28

### Backend

#### Added
- Added named portfolio accounts so a user can maintain a portfolio such as `Dami's portfolio` instead of treating onboarding as the dashboard itself.
- Added `portfolio_activities` persistence for buy/add, transfer, and liquidation events.
- Added portfolio activity APIs for adding purchase lots, listing activity history, renaming the portfolio, and disposing of holdings through transfer or liquidation.
- Added account registration, profile update, password update, subscription summary, and invoice endpoints for signed-in users.
- Added portfolio position trend data based on recent close prices multiplied by held shares for dynamic holding mini charts.
- Added in-process guest symbol registration so guest-local watchlist symbols are warmed by the backend refresh loop after the first guest sync.

#### Changed
- Changed trading onboarding persistence so each new holding row records a buy activity and rolls up into the current position snapshot.
- Changed the background refresh service to read `REFRESH_INTERVAL_SECOND` as well as `REFRESH_INTERVAL_SECONDS`.
- Changed refresh behavior to dedupe active symbols across all persisted users before Yahoo/cache calls, avoiding both per-user duplicate refreshes and whole-market scans.
- Changed the backend refresh loop to actively refresh 5-minute and daily history for the deduped active universe, so watchlist and portfolio mini charts can update while only the venv server is running.

#### Fixed
- Fixed Python 3.9/Pydantic compatibility for newly added account and portfolio request schemas.
- Fixed FastAPI startup compatibility after adding the account and portfolio activity routes.
- Fixed logged-in correlation 500s caused by malformed Yahoo history CSV cache files; corrupt caches are now discarded and refetched or ignored instead of raising a Pandas parser error.
- Fixed the analysis API period validation so the frontend `YTD` chart range can request Yahoo's `ytd` period without being rejected.

## 2026-08-20

### Backend

#### Added
- Added an application-scope workspace refresh service that warms active persisted instruments every 30 seconds without removing the manual `Sync Workspace` action.
- Added `/api/workspace/sync` for scoped watchlist/portfolio refreshes, including guest-local instrument lists sent from the frontend.
- Added RBAC permission groups for account profile, email, security, preferences, subscription, billing, invoices, real-time analysis, and buy/sell chart indicator capabilities.
- Added temp guest JWT session support with analysis/watchlist permissions but no portfolio persistence permission.
- Added Yahoo-backed ticker search at `/api/tickers/search`, returning validated ticker suggestions with market, display label, currency, and company name.
- Added yfinance company-name search candidates to `/api/tickers/search`, so users can search by symbol or company name.
- Added TSX shorthand support during onboarding, so selecting TSX and entering `CCO` resolves to `CCO.TO`.
- Added backend ticker validation before persistence so non-existent symbols are rejected instead of saved.
- Added portfolio-level execution plans for trading holdings.
- Added portfolio metric currency context.
- Added `backend/run_api.sh` to start the FastAPI backend on port `8020`.

#### Changed
- Changed per-symbol news fetching to use Yahoo Finance and Google News queries tied to the selected symbol/company name instead of market-wide generic feeds.
- Changed news sentiment fallback so generic or unmatched headlines no longer influence an instrument score.
- Changed news caching to use a schema version so stale market-wide cache rows are ignored after the symbol-specific filter change.
- Changed portfolio and watchlist sync payloads to include computed watchlist trend data for frontend mini charts.
- Changed backend dev port to `8020` to avoid conflicts with the other local backend.
- Changed ticker search market filtering so alternate listings from unrelated exchanges are not returned for selected TSX/NASDAQ/NYSE filters.
- Changed CORS origin handling so required local frontend origins are merged with configured env origins, preventing stale parent-process env values from blocking the active Next.js dev server.
- Changed trading onboarding persistence so book cost is calculated from `average purchase price * shares` instead of entered manually.
- Confirmed this project uses Python only from `/Users/damidahunsi/stockData/backend/.venv`; no root-level Python venv is required.
- Changed workspace storage identity so unauthenticated guest requests no longer resolve to the local root database user.
- Changed the bootstrapped local root account role from `admin` to `superadmin` to match the system actor model.

#### Fixed
- Fixed FastAPI route annotation compatibility with the Python 3.9 backend venv for `uvicorn app.main:app`.
- Fixed false-positive news matching for short TSX symbols so unrelated TD ETF, AT&T, or broad market headlines are rejected unless they match exchange-qualified ticker/company aliases.
- Fixed RBAC role normalization so legacy `auth_user` maps to the current `user` actor while `guest`, `user`, `admin`, and `superadmin` remain explicit.
- Fixed `backend/.venv` activation metadata so `source .venv/bin/activate` from the backend folder points to `backend/.venv`.
- Fixed `uvicorn app.main:app --reload` startup from inside the backend folder.
- Fixed FastAPI imports so `app.main:app` works when launched from the `backend` directory.
- Fixed left-panel ticker persistence to validate symbols and surface `422` errors for invalid entries.
- Fixed portfolio holdings persistence so entered holdings can appear in the portfolio dashboard with computed allocation, market value, and return/loss metrics.
- Stopped the stale backend process that was running on port `8000` from the old root-venv command.
- Fixed root README startup instructions so backend commands use `backend/.venv`, `app.main:app`, and port `8020` instead of the removed root venv and old port `8000`.
- Fixed logged-in-vs-guest data leakage where guest workspace requests could display the same database-backed instruments as the logged-in root user.
