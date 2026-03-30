from transformers import pipeline
import os

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

messages = [
    "I had a great time tonight, can't wait to see you again!",
    "Yeah, it was fine. See ya around.",
    "I'm actually going to be pretty busy for the next... three years.",
    "You should use more deodorant, just saying...",
]

labels = ["romantic interest", "friend zoned", "polite rejection", "insulting"]

os.system('cls' if os.name == 'nt' else 'clear')

for msg in messages:
    result = classifier(msg, candidate_labels=labels)
    print(f"Text: {msg}")
    print(f"Top Category: {result['labels'][0]} ({result['scores'][0]:.2f} score)\n")