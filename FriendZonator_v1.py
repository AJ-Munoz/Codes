from transformers import pipeline
import os
#pip install transformers[torch] accelerate

# Load a sentiment model
judge = pipeline("sentiment-analysis")

messages = [
    "I had a great time tonight, can't wait to see you again!",
    "Yeah, it was fine. See ya around.",
    "I'm actually going to be pretty busy for the next... three years.",
    "You should use more deodorant, just saying...",
]

os.system('cls' if os.name == 'nt' else 'clear')
for msg in messages:
    result = judge(msg)[0] # E.g. {'label': 'POSITIVE', 'score': 0.95}
    verdict = "LOVE 💖" if result['label'] == 'POSITIVE' else "IT'S OVER 👻"
    print(f"Message: {msg}\nVerdict: {verdict} (Confidence: {result['score']:.2%})\n")