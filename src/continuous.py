import os
import subprocess
import sounddevice as sd
import soundfile as sf
import queue
import vosk
import sys
import json
import threading
from fuzzywuzzy import process as fuzzy_process
from dotenv import load_dotenv
import difflib
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# TTS backend selection (robot/practical)
# Supported: say (macOS built-in), pyttsx3 (offline), none/off
TTS_BACKEND = os.getenv("TTS_BACKEND", "say").lower()
SAY_VOICE = os.getenv("SAY_VOICE", "Zarvox")

# Prevent echo: pause mic capture while Pri is speaking
MIC_PAUSE = threading.Event()
MIC_PAUSE.clear()

def speak_and_pause(message: str, post_delay: float = 0.35):
    """Temporarily pause mic capture while speaking to reduce echo."""
    MIC_PAUSE.set()
    try:
        speak(message)
    finally:
        # Give the audio a moment to finish before resuming mic capture
        import time
        time.sleep(post_delay)
        MIC_PAUSE.clear()

def speak(message: str):
    backend = os.getenv("TTS_BACKEND", "say").lower()
    if backend in ("none", "off", "false", "0"):
        return
    if backend == "pyttsx3":
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 185)
            engine.say(message)
            engine.runAndWait()
            return
        except Exception as e:
            print(f"⚠️ pyttsx3 failed, falling back to say: {e}")
    try:
        subprocess.run(["say", "-v", SAY_VOICE, message])
    except Exception as e:
        print(f"⚠️ macOS say failed: {e}")

def listen_for_stop(event, exit_event):
    try:
        model = vosk.Model(MODEL_PATH)
        rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)

        q = queue.Queue()

        def callback(indata, frames, time, status):
            if status:
                print(f"⚠️ {status}", file=sys.stderr)
            q.put(bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while not event.is_set():
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").lower()
                    if "exit" in text or "exit pri" in text:
                        print("🛑 Exit command detected!")
                        print("👋 Pri says goodbye and powers down.")
                        event.set()
                        exit_event.set()
                    elif text.strip() in ["stop", "enough"] and len(text.strip().split()) == 1:
                        print("🛑 Stop command detected!")
                        print("🤐 Pri politely zips her mouth!")
                        event.set()
                    elif "thanks" in text or "thank you" in text:
                        print("🙏 Polite exit command detected!")
                        print("👋 Pri waves goodbye for now!")
                        event.set()
    except Exception as e:
        print(f"❌ Error in stop listener: {e}")
        print("🥲 Pri feels embarrassed... but ready for your next 'Yo!'")

SAMPLE_RATE = 16000
MODEL_PATH = os.path.expanduser("~/vosk-model-small-en-us-0.15")
AUDIO_DATA_DIR = "data"

def main():
    risk_keywords = ["cut", "knife", "fall", "hurt", "injury", "danger", "blood", "accident", "pain", "wound"]
    stock_aliases = {
        "tesla": "TSLA",
        "apple": "AAPL",
        "nvidia": "NVDA",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "meta": "META",
        "amazon": "AMZN",
        "netflix": "NFLX",
        "intel": "INTC"
    }
    stock_names = list(stock_aliases.keys())
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Vosk model not found at {MODEL_PATH}. Please download and unpack it first.")
        sys.exit(1)

    if not os.path.exists(AUDIO_DATA_DIR):
        os.makedirs(AUDIO_DATA_DIR)

    def speak_greeting():
        greeting_text = "Hello Sujal, how may I assist you today?"
        speak_and_pause(greeting_text, post_delay=0.6)

    if os.getenv("DISABLE_GREETING", "0") != "1":
        print("👋 Starting Pri... Greeting now.")
        speak_greeting()
    else:
        print("👋 Starting Pri... (greeting disabled)")

    model = vosk.Model(MODEL_PATH)
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ {status}", file=sys.stderr)
        # Drop mic frames while Pri is speaking to avoid echo re-trigger
        if MIC_PAUSE.is_set():
            return
        q.put(bytes(indata))

    # --- Helper for fuzzy stock detection ---
    def resolve_stock_from_text(text: str):
        tokens = [t.strip(".,!?\"'()[]{}:").lower() for t in text.split() if t.strip()]
        for t in tokens:
            if t in stock_aliases:
                return t, stock_aliases[t], 100
        match = fuzzy_process.extractOne(" ".join(tokens), stock_names)
        if match:
            name, score = match
            if score >= 80:
                return name, stock_aliases[name], score
        for t in tokens:
            match = fuzzy_process.extractOne(t, stock_names)
            if match:
                name, score = match
                if score >= 85:
                    return name, stock_aliases[name], score
        return None, None, 0

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
        print("🙏 Pri is listening for wake word 'hey'...")

        while True:
            data = q.get()
            joke_detected = False
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower()

                if any(trigger in text for trigger in ["simulate", "stock", "portfolio"]):
                    matched_name, matched_symbol, score = resolve_stock_from_text(text)
                    if matched_symbol:
                        try:
                            announce = (
                                f"Portfolio mode activated for {matched_name.upper()}. "
                                f"Running simulation now."
                            )
                            print(f"🧠 {announce}")
                            speak_and_pause(announce, post_delay=0.6)

                            proc = subprocess.run(
                                ["python", "src/app.py", "simulate-stock", matched_symbol],
                                cwd=os.path.join(os.path.dirname(__file__), ".."),
                                capture_output=True,
                                text=True
                            )

                            if proc.stdout:
                                print(proc.stdout)
                            if proc.stderr:
                                print(proc.stderr, file=sys.stderr)

                            # Only claim success if the command succeeded and produced expected output
                            out = (proc.stdout or "")
                            success = (proc.returncode == 0) and ("Expected Return" in out or "📽️ Video simulation saved" in out or "Monte Carlo" in out)

                            if success:
                                print("🎯 Stock insight delivered. Pri is ready for your next query.")
                            else:
                                print("⚠️ Portfolio simulation did not complete successfully. Please try again.")
                                speak_and_pause("Sorry, I couldn't complete that simulation. Please try again.", post_delay=0.6)

                        except Exception as e:
                            print(f"❌ Error during portfolio simulation trigger: {e}")
                            speak_and_pause("Sorry, something went wrong while running the simulation.", post_delay=0.6)

                        print("🙏 Pri is back to listening for wake word 'hey'...")
                        continue

                # Special Singing Mode trigger
                if (text.strip() == "rap" or text.strip() == "sing"):
                    try:
                        print("🎶 Singing mode activated! Pri is preparing a mini rap...")
                        rap_text = (
                            "Yo, Sujal's on the rise, reaching for the skies, \n"
                            "Ayy, building dreams with code, no disguise. \n"
                            "Yeahh, stepping bold, heart made of gold, \n"
                            "Let's goo, chasing dreams untold. \n"
                            "Ayy, shining bright through every stream, \n"
                            "Yo, Sujal's living the ultimate dream! 🎤✨"
                        )
                        speak_and_pause(rap_text, post_delay=0.6)
                        speak_and_pause("Yo, how was it, my bro? 🎤", post_delay=0.6)
                        print("😎 Pri finished her mini performance!")
                        continue
                    except Exception as e:
                        print(f"❌ Error during singing mode: {e}")

                # Emotional risk detection first
                if any(risk_word in text for risk_word in risk_keywords):
                    speak_and_pause("Please be careful, Sujal.", post_delay=0.6)

                if text:
                    print(f"🔍 Recognized speech: '{text}'")
                    if "joke" in text:
                        joke_detected = True

                # Detect flexible wake words
                if "hey" in text:
                    print("🗣️ Hey! I'm here! Listening to you now...")
                    print("😂 You caught my attention! Let's hear it!")

                    try:
                        # Pause wake-word mic stream while we record the question to avoid echo/re-triggers
                        MIC_PAUSE.set()
                        import time
                        time.sleep(0.35)  # grace period so the user can start speaking

                        print("🎤 Recording... Speak now.")
                        duration = int(os.getenv("QUESTION_MAX_SECONDS", "14"))
                        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
                        sd.wait()
                        time.sleep(0.25)
                        MIC_PAUSE.clear()

                        filename = os.path.join(AUDIO_DATA_DIR, "mic_question.wav")
                        sf.write(filename, audio, SAMPLE_RATE)
                        print(f"✅ Recorded question to {filename}")

                        stop_event = threading.Event()
                        exit_event = threading.Event()
                        stop_listener = threading.Thread(target=listen_for_stop, args=(stop_event, exit_event))
                        stop_listener.start()

                        env = os.environ.copy()
                        p = subprocess.Popen(
                            ["python", "src/app.py", "ask", filename],
                            env=env,
                            cwd=os.path.join(os.path.dirname(__file__), "..")
                        )
                        while p.poll() is None:
                            if stop_event.is_set():
                                p.terminate()
                                print("🛑 Playback interrupted politely.")
                                break

                        if exit_event.is_set():
                            print("👋 Pri is shutting down gracefully. See you soon!")
                            return  # terminate main() cleanly
                        # Otherwise, continue listening
                        print("😄 That was fun! Just say 'Hey' if you need me again!")
                        print("😎 Pri is chilling and waiting for your next 'Hey!' 😎")
                        import time
                        time.sleep(0.5)  # small delay to allow microphone to reset properly
                        print("🙏 Pri is back to listening for wake word 'hey'...")
                        continue  # go back to top of the while loop

                    except Exception as e:
                        print(f"❌ Error during recording or processing: {e}")
                        print("🥲 Pri feels embarrassed... but ready for your next 'Hey!'")

    print("👋 Pri has stopped listening. Goodbye!")
    # sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("👋 Pri says goodbye! See you later!")