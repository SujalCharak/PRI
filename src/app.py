#!/usr/bin/env python3
# src/app.py
# For plotting simulation results
import matplotlib.pyplot as plt
import time
import pandas as pd
import io

"""
Pri: Protected-Retrieval Interface
CLI for speaker-gated QA over audio + text.
"""

# --- Environment and API Key Setup ---
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


import yfinance as yf

# TTS backend selection
import subprocess
TTS_BACKEND = os.getenv("TTS_BACKEND", "say").lower()
SAY_VOICE = os.getenv("SAY_VOICE", "Zarvox")

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

elevenlabs_client = None
stream = None
if TTS_BACKEND == "elevenlabs":
    if not ELEVENLABS_VOICE_ID:
        raise RuntimeError("❗ ELEVENLABS_VOICE_ID not found. Please set it in your .env file.")
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("❗ ELEVENLABS_API_KEY not found. Please set it in your .env file.")
    from elevenlabs.client import ElevenLabs
    from elevenlabs import stream as _stream
    stream = _stream
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

import logging
import click
import soundfile as sf
import sounddevice as sd
import faiss
from resemblyzer import preprocess_wav

import numpy as np
import librosa
import json
from fuzzywuzzy import process as fuzzy_process

PORTFOLIO_KEYWORDS = [
    "stock", "stocks", "portfolio", "investment", "investments", "trading",
    "apple stock", "tesla stock", "share market", "buy stock", "sell stock", "market update"
]

# Expanded stock aliases for broader coverage
STOCK_ALIASES = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
    "netflix": "NFLX",
    "facebook": "META",
    "alphabet": "GOOGL",
    "intel": "INTC"
}

def resolve_stock_symbol_from_prompt(prompt: str):
    """Resolve a ticker from prompt using exact + fuzzy matching on STOCK_ALIASES keys."""
    tokens = [t.strip(".,!?\"'()[]{}:").lower() for t in prompt.split() if t.strip()]

    # Exact match first
    for t in tokens:
        if t in STOCK_ALIASES:
            return STOCK_ALIASES[t], t, 100

    # Fuzzy match each token against known aliases
    best_symbol = None
    best_key = None
    best_score = 0
    keys = list(STOCK_ALIASES.keys())
    for t in tokens:
        match = fuzzy_process.extractOne(t, keys)
        if not match:
            continue
        key, score = match
        if score > best_score:
            best_score = score
            best_key = key
            best_symbol = STOCK_ALIASES[key]

    # Require decent confidence to avoid wrong picks
    if best_symbol and best_score >= 85:
        return best_symbol, best_key, best_score

    return None, None, best_score

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def speak(text: str):
    """Speak text using the configured TTS backend.

    TTS_BACKEND:
      - say: macOS built-in voice (robot-ish, fast, no setup)
      - pyttsx3: offline TTS (cross-platform-ish)
      - elevenlabs: API TTS (requires keys)
      - none/off: disable speaking
    """
    backend = os.getenv("TTS_BACKEND", "say").lower()
    if backend in ("none", "off", "false", "0"):
        return

    # Keeping ElevenLabs available but also default to practical local voices.
    if backend == "elevenlabs":
        if elevenlabs_client is None or stream is None:
            logger.warning("ElevenLabs backend selected but not initialized; falling back to 'say'.")
            backend = "say"
        else:
            try:
                audio = elevenlabs_client.generate(
                    text=text,
                    voice=ELEVENLABS_VOICE_ID,
                    model="eleven_monolingual_v1",
                    voice_settings={
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "speed": 0.78
                    }
                )
                stream(audio)
                return
            except Exception as e:
                logger.warning(f"ElevenLabs TTS failed; falling back to 'say': {e}")
                backend = "say"

    if backend == "pyttsx3":
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            engine.say(text)
            engine.runAndWait()
            return
        except Exception as e:
            logger.warning(f"pyttsx3 TTS failed; falling back to 'say': {e}")
            backend = "say"

    # macOS built-in TTS
    try:
        subprocess.run(["say", "-v", SAY_VOICE, text])
    except Exception as e:
        logger.warning(f"macOS say failed: {e}")


# ─── SENTIMENT ANALYSIS FOR STOCK NEWS (LIVE VIA NEWSAPI) ────────────────────────────────
import requests

def analyze_live_sentiment(stock):
    # Replace with your actual NewsAPI key
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        return f"❗ NEWS_API_KEY not found. Please set it in your .env file."

    url = f"https://newsapi.org/v2/everything?q={stock}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            return f"🔍 No recent news found for {stock.upper()}."

        seen = set()
        headlines = []
        for article in articles:
            title = article["title"]
            if title:
                normalized = title.lower().replace('"', '').replace("'", '').strip()
                if normalized not in seen:
                    headlines.append(title)
                    seen.add(normalized)
            if len(headlines) == 3:
                break
        results = get_sentiment_pipeline()(headlines)

        pos = sum(1 for r in results if r["label"] == "POSITIVE")
        neg = sum(1 for r in results if r["label"] == "NEGATIVE")
        total = len(results)

        if pos > neg:
            summary = f"📊 Sentiment on {stock} is mostly positive ({pos} of {total} headlines)."
        elif neg > pos:
            summary = f"📊 Sentiment on {stock} is mostly negative ({neg} of {total} headlines)."
        else:
            summary = f"📊 Sentiment on {stock} is mixed."

        summary_text = "\n📰 News headlines:\n" + "\n".join(f"- {headline}" for headline in headlines)
        return summary + summary_text

    except Exception as e:
        return f"⚠️ Error fetching news sentiment: {str(e)}"

# ─── LIVE STOCK PRICE FETCHING ────────────────────────────────────────────────
def fetch_stock_price(symbol):
    try:
        df = safe_history(symbol, period="1d", interval="1d", retries=2)
        price = float(df["Close"].iloc[-1])
        return f"💰 {symbol.upper()} is currently trading at ${price:.2f}."
    except Exception:
        cached6 = load_cached_history(symbol, "6mo_1d")
        if cached6 is not None and not cached6.empty:
            price = float(cached6["Close"].iloc[-1])
            return f"💰 {symbol.upper()} is currently trading at ${price:.2f}. (cached)"
        return f"⚠️ I'm having trouble getting the latest data for {symbol.upper()} right now (rate limited)."

# ─── YFINANCE CACHE + RETRY ─────────────────────────

def _get_data_dir() -> str:
    """Return data directory path even if global DATA_DIR is defined later."""
    return globals().get("DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

YF_CACHE_DIR = os.path.join(_get_data_dir(), "yf_cache")
os.makedirs(YF_CACHE_DIR, exist_ok=True)

def _cache_path(symbol: str, tag: str) -> str:
    safe = symbol.upper().replace("/", "_")
    return os.path.join(YF_CACHE_DIR, f"{safe}_{tag}.csv")

def load_cached_history(symbol: str, tag: str, max_age_seconds: int = 6 * 3600):
    path = _cache_path(symbol, tag)
    if not os.path.exists(path):
        return None
    try:
        age = time.time() - os.path.getmtime(path)
        if age > max_age_seconds:
            return None
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
        return df
    except Exception:
        return None

def save_cached_history(symbol: str, tag: str, df: "pd.DataFrame"):
    try:
        df.to_csv(_cache_path(symbol, tag))
    except Exception:
        pass

def safe_history(symbol: str, period: str, interval: str = "1d", retries: int = 3, backoff: float = 1.6):
    tag = f"{period}_{interval}"
    cached = load_cached_history(symbol, tag)

    last_err = None
    for attempt in range(retries):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            if df is not None and not df.empty:
                save_cached_history(symbol, tag, df)
                return df
            last_err = RuntimeError("Empty history returned")
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt)

    if cached is not None and not cached.empty:
        return cached

    raise last_err or RuntimeError("Failed to fetch history")

# ─── CONFIG ────────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "..", "data")
SPEAKERS_FILE  = os.path.join(DATA_DIR, "speakers.json")
PASSAGES_FILE  = os.path.join(DATA_DIR, "passages.json")
INDEX_FILE     = os.path.join(DATA_DIR, "faiss.index")

WHISPER_MODEL  = "tiny.en"          
QA_MODEL       = "distilbert-base-uncased-distilled-squad"

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── GLOBAL INITIALIZATION  ───────────────────────────────────────────
speaker_encoder = None
embed_model = None
qa_pipeline = None
whisper_model = None
sentiment_pipeline = None

def get_speaker_encoder():
    global speaker_encoder
    if speaker_encoder is None:
        from resemblyzer import VoiceEncoder
        speaker_encoder = VoiceEncoder(device="cpu")
        logger.info("Loaded speaker embedding model")
    return speaker_encoder

def get_embed_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
    return embed_model

def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        from transformers import pipeline as hf_pipeline
        qa_pipeline = hf_pipeline("question-answering", model=QA_MODEL)
        logger.info(f"Loaded QA pipeline: {QA_MODEL}")
    return qa_pipeline

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(WHISPER_MODEL)
        logger.info(f"Loaded Whisper model: {WHISPER_MODEL}")
    return whisper_model

def get_sentiment_pipeline():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        from transformers import pipeline as hf_pipeline
        sentiment_pipeline = hf_pipeline("sentiment-analysis")
        logger.info("Loaded sentiment-analysis pipeline")
    return sentiment_pipeline

# ─── DATA BOILERPLATE ───────────────────────────────────────────────────────────

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SPEAKERS_FILE):
        with open(SPEAKERS_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(PASSAGES_FILE):
        with open(PASSAGES_FILE, "w") as f:
            json.dump([], f)
    if not os.path.exists(INDEX_FILE):
        # all-MiniLM-L6-v2 embedding dim is 384
        dim = 384
        idx = faiss.IndexFlatL2(dim)
        faiss.write_index(idx, INDEX_FILE)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_index():
    return faiss.read_index(INDEX_FILE)

def save_index(idx):
    faiss.write_index(idx, INDEX_FILE)

# ─── COMMANDS ──────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Pri CLI entrypoint."""
    ensure_data_dir()
    # load_models()
    # If no subcommand show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command()
@click.argument("name")
@click.argument("wav_path", type=click.Path(exists=True))
def enroll(name, wav_path):
    """Enroll a new speaker from a WAV file."""
    #read audio
    wav = preprocess_wav(wav_path)
    emb = get_speaker_encoder().embed_utterance(wav)
    speakers = load_json(SPEAKERS_FILE)
    speakers[name] = emb.tolist()
    save_json(speakers, SPEAKERS_FILE)
    click.echo(f"✅ Enrolled speaker “{name}”")

@cli.command("add-doc")
@click.argument("text", nargs=-1)
def add_doc(text):
    """Add a text passage (space-separated or quoted)."""
    passage = " ".join(text)
    passages = load_json(PASSAGES_FILE)
    passages.append(passage)
    save_json(passages, PASSAGES_FILE)

    # create index for correct vector dimension
    vec = get_embed_model().encode(passage)
    idx = load_index()
    # If the index dimension doesn't match the embedding lets recreate it
    if hasattr(idx, 'd') and idx.d != len(vec):
        idx = faiss.IndexFlatL2(len(vec))
    # Add the new vector
    idx.add(np.array([vec]))
    save_index(idx)

    click.echo("✅ Passage added to index")

@cli.command()
@click.argument("wav_path", type=click.Path(exists=True))
@click.option("--threshold", default=0.6, show_default=True, help="Speaker similarity threshold")
def ask(wav_path, threshold):
    """Ask a question using speaker-gated audio + your text store."""
    # 1) verify speaker
    wav = preprocess_wav(wav_path)
    emb = get_speaker_encoder().embed_utterance(wav)
    speakers = load_json(SPEAKERS_FILE)
    # naive nearest-neighbor for speaker ID
    best, name = max(
        ((cosine(emb, speakers[n]), n) for n in speakers),
        key=lambda x: x[0],
    )
    logger.warning(f"Speaker similarity: {best:.2f}")
    if best < threshold:
        return click.echo(f"❌ Speaker similarity {best:.2f} below threshold {threshold}")

    # 2) transcribe question audio
    segments, info = get_whisper_model().transcribe(wav_path, beam_size=5)
    q = "".join(segment.text for segment in segments).strip()
    logger.info(f"Transcribed question: {q}")

    print(f"📜 Question: {q}")

    # 3) retrieve context
    idx      = load_index()
    passages = load_json(PASSAGES_FILE)
    # Handle FAISS index dimension mismatch
    qvec = get_embed_model().encode([q])[0]
    if hasattr(idx, 'd') and idx.d != len(qvec):
        # Reset index due to dimension mismatch
        new_idx = faiss.IndexFlatL2(len(qvec))
        save_index(new_idx)
        return click.echo("⚠️ Index dimension mismatch. Reset FAISS index. Please re-add documents with `add-doc`.")
    if not passages:
        return click.echo("⚠️ No passages available. Use `add-doc` first.")
    
    D, I     = idx.search(qvec.reshape(1, -1), k=3)
    context  = " ".join(passages[i] for i in I[0])

    print("📚 Context: In Sujal I believe.")

    tone = "neutral"  # will be updated with tone detection later
    answer = generate_gpt_response(q, tone=tone)
    click.echo(f"🧠 Pri Answer ({name}): {answer}")
    speak(answer)

    # If the prompt or answer relates to a joke lets make Pri laugh naturally
    if "joke" in q.lower() or "😂" in answer.lower() or "laugh" in q.lower():
        speak("That was funny! Hehe.")


def generate_gpt_response(prompt, tone="neutral"):
    # Auto-trigger portfolio mode
    # if any(word in prompt.lower() for word in PORTFOLIO_KEYWORDS):
    #     click.echo("📈 Auto-triggered Portfolio Mode based on prompt.")
    #     speak("Portfolio mode activated!")

    # Expanded simulation trigger keywords
    simulation_triggers = ["simulate", "forecast", "projection", "monte carlo", "analyze", "run analysis"]
    if any(word in prompt.lower() for word in simulation_triggers):
        symbol, matched_key, match_score = resolve_stock_symbol_from_prompt(prompt)
        if symbol:
            response_lines = []
            response_lines.append("Portfolio mode activated. Let me check the stock details for you.")
            # Only include price_info if it will not be repeated in the simulation result
            price_info = None  # Preventing duplication and the simulation includes trading price
            # Detect forecast period and risk level
            days = 30
            trials = 1000
            if "15 day" in prompt.lower():
                days = 15
            elif "60 day" in prompt.lower():
                days = 60
            if "low risk" in prompt.lower():
                trials = 500
            elif "high risk" in prompt.lower() or "more precision" in prompt.lower():
                trials = 2000

            result = monte_carlo_simulation(symbol, days=days, trials=trials)
            response_lines.append(result)
            full_response = "\n".join(response_lines)
            click.echo(full_response)
            speak(full_response)
            return full_response  # prevent fallback for gpt/gemini
        else:
            # Trigger words present but no recognizable ticker
            msg = "⚠️ I heard a simulation request, but I couldn't recognize the company name. Please say the stock name again (e.g., 'Meta', 'Apple', 'Tesla')."
            click.echo(msg)
            speak(msg)
            return msg

    # Updated stock ticker detection using aliases which  always returns both price and sentiment
    symbol, matched_key, match_score = resolve_stock_symbol_from_prompt(prompt)
    if symbol:
        price_info = fetch_stock_price(symbol)
        sentiment_summary = analyze_live_sentiment(symbol)
        combined_response = f"{price_info}\n{sentiment_summary}"
        click.echo(combined_response)
        speak(combined_response)
        return combined_response  # Skip GPT, handled with data

    if "rap" in prompt.lower() or "sing" in prompt.lower():
        system_prompt = """You are Pri, Sujal's AI companion. 
You are a rapper and singer when asked. 
When Sujal asks you to rap or sing, generate a short, fun, rhythmic rap with exactly 6 lines maximum.
Use slang words like "yo", "ayy", "yeahh", "let's goo" where appropriate.
Keep it energetic, casual, and cool!"""
    else:
        system_prompt = f"""You are Pri, Sujal's personal assistant. 
You are helpful, witty, slightly sarcastic, and emotionally intelligent.
If you are unsure what Sujal meant, politely ask for clarification. 
You reflect the user's tone: {tone}."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Gemini-only LLM 
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            return "⚠️ GEMINI_API_KEY not set. Please set it in your .env to use Gemini."

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)

        combined = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{prompt}"
        resp = client.models.generate_content(model=model_name, contents=combined)
        content = (resp.text or "").strip()

        if not content:
            return "⚠️ Gemini returned an empty response. Please try again."

    except Exception as e:
        logger.exception(f"Gemini request failed: {e}")
        return f"⚠️ Gemini request failed: {e}"

    # If it's a rap/singing response, append "Yo, how was it?" at the end
    if "rap" in prompt.lower() or "sing" in prompt.lower():
        content += "\n\nYo, how was it? 🎤"

    return content


# ─── KIM'S PORTFOLIO BUILDER ─────────────────────────────────────────────────
def build_portfolio_for_kim():
    """
    Kim is a 25-year-old professional focused on long-term, stable growth.
    This portfolio favors large-cap, well-managed, and diversified companies from the S&P 400 with strong historical performance.
    """
    stable_growth_stocks = ["MSFT", "AAPL", "GOOGL", "JNJ", "NVDA", "PG"]
    click.echo("🧾 Kim's Long-Term Growth Portfolio:")
    descriptions = {
        "MSFT": "Microsoft – Tech giant with consistent earnings and cloud leadership.",
        "AAPL": "Apple – Reliable innovation and brand strength in consumer tech.",
        "GOOGL": "Alphabet – Strong fundamentals and growth from AI and search.",
        "JNJ": "Johnson & Johnson – Diversified healthcare with strong management.",
        "NVDA": "NVIDIA – Leading semiconductor company with growth in AI and gaming.",
        "PG": "Procter & Gamble – Consumer staples with steady cash flow and resilience."
    }
    for stock in stable_growth_stocks:
        summary = fetch_stock_price(stock)
        sentiment = analyze_live_sentiment(stock)
        click.echo(f"\n📌 {descriptions[stock]}")
        click.echo(summary)
        click.echo(sentiment)
        speak(summary)
        speak(sentiment)

@cli.command("ask-mic")
@click.pass_context
@click.option("--duration", default=5, show_default=True, help="Recording duration in seconds")
@click.option("--threshold", default=0.6, show_default=True, help="Speaker similarity threshold")
def ask_mic(ctx, duration, threshold):
    """
    Record audio from the microphone and immediately ask the question.
    """
    samplerate = 16000
    click.echo(f"🎙️ Recording {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    # Save to a temporary file in data directory
    mic_wav_path = os.path.join(DATA_DIR, "mic_question.wav")
    sf.write(mic_wav_path, audio, samplerate)
    click.echo("✅ Recorded. Processing your question...")
    # Delegate to the existing ask command
    ctx.invoke(ask, wav_path=mic_wav_path, threshold=threshold)


# ─── TEXT-ONLY QUESTION COMMAND ──────────────────────────────────────────────

@cli.command("ask-text")
@click.argument("question", nargs=-1)
def ask_text(question):
    """
    Ask a question by typing instead of speaking.
    """
    q = " ".join(question)
    click.echo(f"📜 Typed Question: {q}")

    answer = generate_gpt_response(q)
    click.echo(f"🧠 GPT Answer (text mode): {answer}")


@cli.command("simulate-stock")
@click.argument("symbol")
@click.option("--days", default=30, show_default=True, help="Forecast period in days (default 30)")
@click.option("--trials", default=1000, show_default=True, help="Number of simulation trials (default 1000)")
def simulate_stock(symbol, days, trials):
    """
    Simulate stock future price using Monte Carlo with custom forecast period and precision.
    """
    result = monte_carlo_simulation(symbol.upper(), days=days, trials=trials)
    click.echo(result)
    speak(result)


def monte_carlo_simulation(symbol, days=30, trials=1000):
    try:
        df = safe_history(symbol, period="6mo", interval="1d", retries=3)
        if df.empty:
            return f"⚠️ Not enough historical data for {symbol}."

        log_returns = np.log(1 + df["Close"].pct_change().dropna())
        mu = log_returns.mean()
        sigma = log_returns.std()

        simulated_returns = np.random.normal(mu, sigma, (trials, days))
        price_paths = df["Close"].iloc[-1] * np.exp(simulated_returns.cumsum(axis=1))
        final_prices = price_paths[:, -1]

        expected_return = np.mean((final_prices - df["Close"].iloc[-1]) / df["Close"].iloc[-1]) * 100
        min_price = np.min(final_prices)
        max_price = np.max(final_prices)

        explanation = f"{symbol.upper()} is trading at ${df['Close'].iloc[-1]:.2f}.\n"
        explanation += f"I ran a {days}-day forecast using {trials} scenarios.\n"
        explanation += f"The expected return over this period is approximately {expected_return:.2f}%.\n"
        explanation += f"Prices could vary between ${min_price:.2f} and ${max_price:.2f}, based on recent market trends."

        # meaning of risk level based on number of trials
        if trials <= 600:
            explanation += "\nI used a smaller number of simulations for quicker results, which may reduce stability."
        elif trials >= 2000:
            explanation += "\nI used a large number of simulations to increase precision."
        else:
            explanation += "\nI balanced speed and accuracy using a medium number of simulations."

        # Plot simulation
        plt.figure(figsize=(10, 6))
        plt.plot(price_paths.T, color='blue', alpha=0.1)
        plt.title(f"Monte Carlo Simulation for {symbol} ({days} days)")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.grid(True)

        plot_path = os.path.join(DATA_DIR, f"{symbol}_simulation.png")
        plt.savefig(plot_path)
        plt.close()

        # frame-by-frame plots for video
        frame_dir = os.path.join(DATA_DIR, f"{symbol}_frames")
        os.makedirs(frame_dir, exist_ok=True)

        for day in range(days):
            plt.figure(figsize=(10, 6))
            plt.plot(price_paths[:, :day+1].T, color='blue', alpha=0.1)
            plt.title(f"{symbol} Simulation - Day {day+1}")
            plt.xlabel("Day")
            plt.ylabel("Price")
            plt.grid(True)
            frame_file = os.path.join(frame_dir, f"frame_{day:03d}.png")
            plt.savefig(frame_file)
            plt.close()

        # video from frames using ffmpeg 
        video_output = os.path.join(DATA_DIR, f"{symbol}_simulation.webm")
        try:
            os.system(f"ffmpeg -y -framerate 10 -i {frame_dir}/frame_%03d.png {video_output}")
            explanation += f"\n📽️ Video simulation saved: {video_output}"
        except Exception as e:
            explanation += f"\n⚠️ Failed to generate video: {str(e)}"

        # Opens the video automatically
        try:
            if os.name == 'posix':
                os.system(f"open {video_output}")  # macOS
            elif os.name == 'nt':
                os.startfile(video_output)  # Windows
            else:
                os.system(f"xdg-open {video_output}")  # Linux
        except Exception:
            pass

        # Investment suggestion based on expected return and sentiment
        sentiment_summary = analyze_live_sentiment(symbol)
        explanation += f"\n{sentiment_summary}"

        # Determine tone of sentiment
        sentiment_lower = sentiment_summary.lower()
        if "mostly positive" in sentiment_lower:
            sentiment_score = 1
        elif "mostly negative" in sentiment_lower:
            sentiment_score = -1
        else:
            sentiment_score = 0

        # Combining return and sentiment for final recommendation
        if expected_return > 5 and sentiment_score >= 0:
            suggestion = "✅ Entry Signal: Simulation shows favorable returns and sentiment is positive. Consider investing."
        elif expected_return < -5 and sentiment_score <= 0:
            suggestion = "❌ Exit Signal: Simulation predicts decline and sentiment is negative. Consider exiting or avoiding."
        else:
            suggestion = "⏸️ Hold Signal: Simulation and sentiment are inconclusive. Monitor the market before making a move."

        explanation += f"\n{suggestion}"

        return explanation
    except Exception as e:
        import traceback
        return f"⚠️ Simulation failed for {symbol}:\n{traceback.format_exc()}"

if __name__ == "__main__":
    cli()
