from transformers import pipeline
import os, json

# --- Load 4 specialist models ---
judge_cardiff = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
judge_emotion = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions")
judge_sarcasm = pipeline("sentiment-analysis", model="helinivan/english-sarcasm-detector")
judge_toxic = pipeline("sentiment-analysis", model="unitary/toxic-bert")

# --- Reasoning model ---
reasoner = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    pad_token_id=50256
)

def final_judgment(msg):
    c = judge_cardiff(msg)[0]
    e = judge_emotion(msg)[0]
    s = judge_sarcasm(msg)[0]
    t = judge_toxic(msg)[0]

    signal_block = {
        "sentiment": c,
        "emotion": e,
        "sarcasm": s,
        "toxic": t
    }

    prompt = f"""
You are a relationship expert. Interpret the message:

Message: "{msg}"

Here are analytic signals from multiple models:
{json.dumps(signal_block, indent=2)}

Using these signals together, classify the sender's intention as one of:
- romantic interest
- friend zone
- polite rejection
- insulting
- unclear

Give a SHORT final label only.
"""

    out = reasoner(prompt, max_new_tokens=50)[0]["generated_text"]
    cleaned = out.split(prompt)[-1].strip()
    return cleaned

# --- Test messages ---
messages = [
    "I had a great time tonight, can't wait to see you again!",
    "Yeah, it was fine. See ya around.",
    "I'm actually going to be pretty busy for the next... three years.",
    "You should use more deodorant, just saying...",
]

for m in messages:
    print("\nMessage:", m)
    print("FINAL:", final_judgment(m))