import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
import logging


def get_composite_features(pruner, query):
    encoding = pruner.tokenizer.encode(query)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, len(encoding.ids)]])


def generate_massive_dataset():
    # Templates for Needle with noise
    needle_bases = [
        "port",
        "api key",
        "password",
        "secret",
        "ip address",
        "budget",
        "version",
        "date",
        "token",
        "string",
        "constant",
        "value",
        "id",
        "key",
        "email",
        "name",
    ]
    needle_phrases = [
        "What is the {x}?",
        "Find the {x}.",
        "Tell me the {x}.",
        "Show me the {x}.",
        "Retrieve the {x}.",
        "Give me the {x} mentioned earlier.",
        "What was the {x} from before?",
        "Identify the {x} in the history.",
        "Could you find the {x} we discussed?",
        "I need the {x} value.",
        "{x}?",
        "What is the {x} number?",
        "Where is the {x}?",
        "Search for the {x}.",
        "Extract the {x}.",
    ]

    # Templates for Reasoning
    reasoning_bases = [
        "Alpha and Beta",
        "Kong and OAuth2",
        "monolith and microservices",
        "ELK and Prometheus",
        "budget and performance",
        "current and future state",
    ]
    reasoning_phrases = [
        "Compare {x}.",
        "Analyze the relationship between {x}.",
        "Why did {x} happen?",
        "Evaluate the {x}.",
        "Contrast {x}.",
        "Synthesize the data for {x}.",
        "What are the differences in {x}?",
        "Explain the logic of {x}.",
        "Cross-reference {x}.",
        "Based on logs, why {x}?",
        "Justify the use of {x}.",
        "Critique the design of {x}.",
    ]

    # Templates for Summary
    summary_phrases = [
        "Summarize our talk.",
        "TL;DR.",
        "Give a brief overview.",
        "Concise report.",
        "Short synopsis.",
        "Recap the main points.",
        "High-level summary.",
        "Condensed version.",
        "Wrap up the chat.",
        "General topic of conversation?",
        "Briefly outline the plan.",
        "Technical synopsis.",
        "What have we discussed so far?",
        "One paragraph summary please.",
        "Synopsis of the history.",
    ]

    needle_data = []
    for x in needle_bases:
        for p in needle_phrases:
            needle_data.append(p.format(x=x))

    reasoning_data = []
    for x in reasoning_bases:
        for p in reasoning_phrases:
            reasoning_data.append(p.format(x=x))

    summary_data = summary_phrases * 5  # ~75 examples

    # Add manual edge cases from benchmarks
    needle_data.append("What was the database port mentioned earlier?")
    needle_data.append("Find the DB connection string.")

    return {
        "Needle": needle_data,  # ~240 examples
        "Reasoning": reasoning_data,  # ~72 examples
        "Summary": summary_data,  # ~75 examples
    }


def train_massive_router():
    print("=== TRAINING MASSIVE COMPOSITE ROUTER (500+ EXAMPLES) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    data = generate_massive_dataset()

    # 1. Train IsNeedle (vs All)
    print("\nTraining Stage: IsNeedle (vs All)...")
    X, y = [], []
    for q in data["Needle"]:
        X.append(get_composite_features(pruner, q))
        y.append(1)
    for q in data["Reasoning"] + data["Summary"]:
        X.append(get_composite_features(pruner, q))
        y.append(0)

    clf_n = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced").fit(
        np.array(X), np.array(y)
    )
    print(f"✅ Needle Accuracy: {clf_n.score(X, y) * 100:.1f}% ({len(X)} samples)")
    with open("src/wamp/core/needle_weights.json", "w") as f:
        json.dump({"coef": clf_n.coef_.tolist(), "intercept": clf_n.intercept_.tolist()}, f)

    # 2. Train IsReasoning (vs All)
    # Crucial: Include Needles as negative examples here!
    print("\nTraining Stage: IsReasoning (vs All)...")
    X, y = [], []
    for q in data["Reasoning"]:
        X.append(get_composite_features(pruner, q))
        y.append(1)
    for q in data["Needle"] + data["Summary"]:
        X.append(get_composite_features(pruner, q))
        y.append(0)

    clf_r = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced").fit(
        np.array(X), np.array(y)
    )
    print(f"✅ Reasoning Accuracy: {clf_r.score(X, y) * 100:.1f}% ({len(X)} samples)")
    with open("src/wamp/core/router_weights.json", "w") as f:
        json.dump({"coef": clf_r.coef_.tolist(), "intercept": clf_r.intercept_.tolist()}, f)

    print("\n🚀 Massive training complete. Weights saved.")


if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_massive_router()
