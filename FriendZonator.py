"""
FriendZonator 💘  - one script, all the judges.

Modes
  quick : zero-shot classifier only (1 model, fast)
  full  : zero-shot + sentiment + emotion + sarcasm + toxicity (5 models, default)

Usage
  python FriendZonator.py                         # judge the built-in examples
  python FriendZonator.py --mode quick
  python FriendZonator.py "see you never" "ok"    # judge your own messages
  python FriendZonator.py -i                      # interactive, Ctrl+C to quit
  python FriendZonator.py -v                      # show every model's raw signals

Install
  pip install "transformers[torch]" accelerate
"""
import argparse
from functools import lru_cache

from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()  # hide the wall of HF warnings

# --------------------------------------------------------------------------
# Models (loaded lazily, only when a mode actually needs them)
# --------------------------------------------------------------------------
MODELS = {
    "zero_shot": ("zero-shot-classification", "facebook/bart-large-mnli"),
    "sentiment": ("text-classification", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    "emotion":   ("text-classification", "SamLowe/roberta-base-go_emotions"),
    "sarcasm":   ("text-classification", "helinivan/english-sarcasm-detector"),
    "toxic":     ("text-classification", "unitary/toxic-bert"),
}


@lru_cache(maxsize=None)
def judge(name):
    task, model = MODELS[name]
    print(f"  ...summoning the {name} judge ({model})")
    return pipeline(task, model=model)


def all_scores(name, msg):
    """Run a text-classification model and return {label: score} for EVERY label."""
    out = judge(name)(msg, top_k=None)
    if out and isinstance(out[0], list):  # some transformers versions nest the list
        out = out[0]
    return {d["label"]: d["score"] for d in out}


# --------------------------------------------------------------------------
# Categories and verdicts
# --------------------------------------------------------------------------
# label -> (hypothesis text for zero-shot, verdict shown to the user)
CATEGORIES = {
    "romantic":  ("romantic interest in the other person", "LOVE 💖💘 (Mutual vibes detected!)"),
    "friend":    ("purely friendly, platonic feelings",    "FRIENDZONE 🤝 (It's not you, it's... well, yeah.)"),
    "rejection": ("a polite rejection",                    "GHOSTING IMMINENT 👻"),
    "insult":    ("an insult",                             "ABSOLUTELY OVER 💀"),
}
HYPOTHESIS_TEMPLATE = "This message expresses {}."

LOVE_EMOTIONS = {"love", "admiration", "excitement", "joy", "desire", "caring"}
COLD_EMOTIONS = {"annoyance", "disapproval", "disappointment", "disgust", "anger"}

# How much each specialist nudges the zero-shot base scores. Tune freely.
WEIGHTS = {
    "sentiment": 0.5,
    "emotion": 0.5,
    "toxic": 1.0,
}
SARCASM_THRESHOLD = 0.70
THREAT_THRESHOLD = 0.50


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------
def zero_shot(msg):
    hypotheses = {hyp: key for key, (hyp, _) in CATEGORIES.items()}
    r = judge("zero_shot")(msg, candidate_labels=list(hypotheses),
                           hypothesis_template=HYPOTHESIS_TEMPLATE)
    return {hypotheses[l]: s for l, s in zip(r["labels"], r["scores"])}


def normalize(scores):
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def judge_message(msg, mode="full"):
    """Return (scores per category, flags, raw signals)."""
    scores = zero_shot(msg)
    flags, raw = [], {"zero_shot": dict(scores)}

    if mode == "quick":
        return scores, flags, raw

    sent = all_scores("sentiment", msg)  # positive / neutral / negative
    emo = all_scores("emotion", msg)     # 28 GoEmotions labels
    sarc = all_scores("sarcasm", msg)    # LABEL_0 = sincere, LABEL_1 = sarcastic
    tox = all_scores("toxic", msg)       # toxic, insult, threat, ... (independent scores)
    raw.update(sentiment=sent, sarcasm=sarc, toxic=tox,
               emotion=dict(sorted(emo.items(), key=lambda kv: -kv[1])[:5]))

    love = sum(emo.get(e, 0) for e in LOVE_EMOTIONS)
    cold = sum(emo.get(e, 0) for e in COLD_EMOTIONS)
    toxicity = max(tox.get("toxic", 0), tox.get("insult", 0))
    sarcasm = sarc.get("LABEL_1", 0)

    w = WEIGHTS
    scores["romantic"]  += w["sentiment"] * sent.get("positive", 0) + w["emotion"] * love
    scores["friend"]    += w["sentiment"] * sent.get("neutral", 0)
    scores["rejection"] += w["sentiment"] * sent.get("negative", 0) + w["emotion"] * cold
    scores["insult"]    += w["toxic"] * toxicity

    # Sarcastic "affection" isn't affection: move the romance to rejection.
    if sarcasm > SARCASM_THRESHOLD:
        scores["rejection"] += scores["romantic"]
        scores["romantic"] = 0.0
        flags.append(f"IT'S A TRAP 🎭 sarcasm detected ({sarcasm:.0%})")

    if toxicity > 0.5:
        flags.append(f"Toxicity detected ({toxicity:.0%})")
    if tox.get("threat", 0) > THREAT_THRESHOLD:
        flags.append("Restraining order pending 👮🚔")

    return normalize(scores), flags, raw


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def bar(p, width=20):
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def report(msg, mode, verbose):
    scores, flags, raw = judge_message(msg, mode)
    winner = max(scores, key=scores.get)

    print(f"\n--- \"{msg}\" ---")
    for key, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<10} {bar(p)} {p:6.1%}")
    for f in flags:
        print(f"  [ALERT] {f}")
    if verbose:
        for name, sig in raw.items():
            pretty = ", ".join(f"{k}={v:.2f}" for k, v in sig.items())
            print(f"  [{name}] {pretty}")
    print(f"  >> VERDICT: {CATEGORIES[winner][1]}  ({scores[winner]:.0%})")


EXAMPLES = [
    "I had a great time tonight, can't wait to see you again!",
    "Yeah, it was fine. See ya around.",
    "I'm actually going to be pretty busy for the next... three years.",
    "You should use more deodorant, just saying...",
]


def main():
    ap = argparse.ArgumentParser(description="Judge the romantic fate of a text message.")
    ap.add_argument("messages", nargs="*", help="messages to judge (default: built-in examples)")
    ap.add_argument("-m", "--mode", choices=["quick", "full"], default="full")
    ap.add_argument("-i", "--interactive", action="store_true", help="type messages one by one")
    ap.add_argument("-v", "--verbose", action="store_true", help="show raw model signals")
    args = ap.parse_args()

    if args.interactive:
        print("Paste a message (Ctrl+C to quit).")
        try:
            while True:
                msg = input("\n💌 > ").strip()
                if msg:
                    report(msg, args.mode, args.verbose)
        except (KeyboardInterrupt, EOFError):
            print("\nGood luck out there. 🫡")
        return

    for msg in args.messages or EXAMPLES:
        report(msg, args.mode, args.verbose)


if __name__ == "__main__":
    main()
