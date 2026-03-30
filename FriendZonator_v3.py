from transformers import pipeline
import os

# --- Load Models ---
judge_cardiff = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

judge_sam = pipeline(
    "sentiment-analysis", 
    model="SamLowe/roberta-base-go_emotions"
)

judge_sarcastic = pipeline(
    "sentiment-analysis",
    model="helinivan/english-sarcasm-detector"
)

judge_toxic = pipeline(
    "sentiment-analysis",
    model="unitary/toxic-bert"
)

messages = [
    "I had a great time tonight, can't wait to see you again!",
    "Yeah, it was fine. See ya around.",
    "I'm actually going to be pretty busy for the next... three years.",
    "You should use more deodorant, just saying...",
]

os.system("cls" if os.name == "nt" else "clear")

for msg in messages:
    print(f"\n--- ANALYZING: \"{msg}\" ---")

    # Run all models
    c = judge_cardiff(msg)[0]      # positive, neutral, negative
    s = judge_sam(msg)[0]          # emotion label
    h = judge_sarcastic(msg)[0]    # LABEL_0 = not sarcasm, LABEL_1 = sarcasm
    t = judge_toxic(msg)[0]        # toxic / non-toxic

    # --- Verdict Logic for Cardiff ---
    cardiff_map = {
        "positive": "LOVE",
        "neutral": "FRIENDZONE",
        "negative": "OVER"
    }
    c_verdict = cardiff_map.get(c["label"], "UNKNOWN")

    # --- Verdict Logic for SamLowe / GoEmotions ---
    love_emotions = {"love", "admiration", "excitement", "joy"}
    neutralish = {"neutral", "approval", "realization", "gratitude"}

    if s["label"] in love_emotions:
        s_verdict = "LOVE"
    elif s["label"] in neutralish:
        s_verdict = "FRIENDZONE"
    else:
        s_verdict = "OVER"

    # --- Sarcasm / Toxicity Flags ---
    is_sarcastic = (h["label"] == "LABEL_1" and h["score"] > 0.70)
    is_toxic = (t["label"] == "toxic" and t["score"] > 0.50)

    # --- Final Verdict Logic ---
    if is_toxic:
        final = "ABSOLUTELY OVER 💀 (Restraining order pending 👮🚔)"
    elif is_sarcastic:
        final = "IT'S A TRAP 🎭 (Sarcasm detected!)"
    elif c_verdict == "LOVE" and s_verdict == "LOVE":
        final = "LOVE 💖💘 (Mutual vibes detected!)"
    elif "FRIENDZONE" in {c_verdict, s_verdict}:
        final = "FRIENDZONE 🤝 (It's not you, it's... well, yeah.)"
    else:
        final = "GHOSTING IMMINENT 👻"

    # --- Print Results ---
    print(f"  [Cardiff]    {c_verdict:<11}  ({c['score']:.1%})")
    print(f"  [GoEmotions] {s_verdict:<11}  ({s['label']}, {s['score']:.1%})")

    if is_sarcastic:
        print(f"  [ALERT]      Sarcasm detected!  ({h['score']:.1%})")

    if is_toxic:
        print(f"  [ALERT]      Toxicity detected! ({t['score']:.1%})")

    print(f"  >> FINAL VERDICT: {final}")