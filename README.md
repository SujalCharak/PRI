# PRI Voice Agent (Portfolio Assistant)

Voice-enabled assistant that can:
- simulate stock price paths (Monte Carlo / GBM)
- summarize news sentiment (optional)
- generate a simple Entry/Hold/Exit suggestion
- create a simulation video (webm) when ffmpeg is available

## Setup
1) Copy env template:
   cp .env.example .env
2) Fill `GEMINI_API_KEY` (required) and `NEWS_API_KEY` (optional)
3) Run:
   python src/continuous.py

## CLI
python src/app.py simulate-stock AAPL --days 30 --trials 500
python src/app.py ask-text "Summarize Tesla sentiment"
