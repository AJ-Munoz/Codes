from transformers import pipeline
import os

# Load Sentiment Models
judge_cardiff = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

judge_sam = pipeline(
    "sentiment-analysis", 
    model="SamLowe/roberta-base-go_emotions"
)

messages = [
    "I had a great time tonight, can't wait to see you again!",
    "Yeah, it was fine. See ya around.",
    "I'm actually going to be pretty busy for the next... three years.",
    "You should use more deodorant, just saying...",
]

os.system('cls' if os.name == 'nt' else 'clear')
for msg in messages:
    print(f"\nJudging: \"{msg}\" 🤔")

    # --- 1. Cardiff NLP ---
    c = judge_cardiff(msg)[0]  # {label: 'positive'|'neutral'|'negative', score: float}

    if c['label'] == 'positive':
        c_verdict = "LOVE 💖"
    elif c['label'] == 'neutral':
        c_verdict = "FRIENDZONE 🤝"
    else:
        c_verdict = "OVER 👻"

    # --- 2. Sam Lowe (GoEmotions) ---
    s = judge_sam(msg)[0]  # {label: emotion_name, score: float}

    love_emotions = ["love", "admiration", "excitement", "joy"]
    neutralish = ["neutral", "approval", "realization"]
    
    if s["label"] in love_emotions:
        s_verdict = "LOVE 💖"
    elif s["label"] in neutralish:
        s_verdict = "FRIENDZONE 🤝"
    else:
        s_verdict = "OVER 👻"

    print(f"Cardiff says:   {c_verdict} ({c['score']:.1%})")
    print(f"Sam Lowe says:  {s_verdict} ({s['label']}, {s['score']:.1%})")