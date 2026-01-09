# PRI Voice Agent (Portfolio Assistant)

A practical **voice-enabled portfolio assistant** that can:
- Run **Monte Carlo / GBM** simulations for stock price paths
- Optionally summarize **news sentiment** (via NewsAPI)
- Provide a simple **Entry / Hold / Exit** suggestion
- Generate a **simulation video** (`.webm`) if `ffmpeg` is available

> **Scope note:** This repository is for the **PRI Voice Agent** (portfolio assistant).  
> It is **not associated with PRI OS**, the separate self-optimizing operating system project.

> **Disclaimer:** This project is for educational and demo purposes only and is **not financial advice**.

---

## Contents
- [Demo / Presentation](#demo--presentation)
- [Features](#features)
- [How it Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [1) Create Environment](#1-create-environment)
  - [2) Install Dependencies](#2-install-dependencies)
  - [3) Configure Environment Variables](#3-configure-environment-variables)
- [Run](#run)
  - [Voice Mode](#voice-mode)
  - [CLI Mode](#cli-mode)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Demo / Presentation
- Project deck: `docs/Portfolio-PRI.pdf`

---

## Features

### ✅ Voice-first interaction
- Wake word: **“hey”**
- Records your question for a configurable time window
- Text-to-speech via a **practical robot voice** (macOS `say` by default)

### ✅ Portfolio simulation (GBM / Monte Carlo)
- Uses recent historical price data to estimate drift + volatility
- Simulates multiple price paths over a horizon (e.g., 30–60 days)
- Outputs:
  - Expected return estimate
  - Range of possible prices (min/max from simulation endpoints)

### ✅ Entry / Hold / Exit signal
Simple decision logic combining:
- Expected return (from simulations)
- News sentiment (optional)

### ✅ Video simulation output (optional)
- Produces frame-by-frame plots
- Converts them into a video using `ffmpeg`
- Saves video to `data/<TICKER>_simulation.webm`

---

## How it Works (High-level)

1. **Wake word detection** (voice mode)
2. **Speech recording** for a fixed duration
3. **Text intent parsing**
   - portfolio mode triggers: simulate/stock/portfolio
   - fuzzy matching helps map misheard names (e.g. “matter” → “meta”)
4. **Market data fetch**
   - primary: `yfinance`
   - fallback logic can use cached data (if available)
5. **Simulation**
   - GBM-based Monte Carlo runs `trials × days`
6. **Sentiment (optional)**
   - fetch headlines via NewsAPI
   - classify sentiment
7. **Decision signal**
   - Entry / Hold / Exit
8. **Outputs**
   - summary text
   - plot + optional video

---

## Project Structure

```text
PRI/
├─ src/
│  ├─ app.py              # Core logic: simulation, sentiment, CLI commands
│  └─ continuous.py       # Voice mode: wake word, recording, routing to app.py
├─ docs/
│  └─ Portfolio-PRI.pdf   # Presentation deck
├─ .env.example           # Environment variable template (no secrets)
├─ .gitignore
└─ README.md
