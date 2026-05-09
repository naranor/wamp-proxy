import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import json
import os
import logging


def audit_modernbert_onnx():
    print("=== AUDIT: ModernBERT-BASE ONNX ROUTER ===")
    model_dir = "./model_modernbert_onnx"

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 2. Load ONNX Session (Quantized)
    model_path = os.path.join(model_dir, "model_quantized.onnx")
    session = ort.InferenceSession(model_path)

    # 3. Load LogReg Head
    with open(os.path.join(model_dir, "router_weights_setfit.json"), "r") as f:
        head = json.load(f)
    coef = np.array(head["coef"])
    intercept = np.array(head["intercept"])
    label_map = {v: k for k, v in head["label_map"].items()}

    test_queries = [
        "What is the database port?",  # Expected: Needle
        "Compare Alpha and Beta projects.",  # Expected: Reasoning
        "Give me a summary of the chat.",  # Expected: Summary
        "What was the database port mentioned earlier?",  # Needle
    ]

    print(f"{'Query':<45} | {'Summary':<8} | {'Needle':<8} | {'Reason':<8} | {'Winner'}")
    print("-" * 95)

    for query in test_queries:
        inputs = tokenizer(query, return_tensors="np")
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        outputs = session.run(None, onnx_inputs)

        # Last hidden state pooling (outputs[0])
        # ModernBERT has different output order sometimes, let's verify
        last_hidden = outputs[0]
        embeddings = np.mean(last_hidden, axis=1)  # Mean pooling

        # LogReg Prediction
        scores = np.dot(embeddings, coef.T) + intercept
        exp_scores = np.exp(scores - np.max(scores))
        probs = (exp_scores / exp_scores.sum())[0]

        winner_idx = np.argmax(probs)
        winner_label = label_map[winner_idx]

        print(
            f"{query[:43]:<45} | {probs[0]:.3f}   | {probs[1]:.3f}   | {probs[2]:.3f}   | {winner_label}"
        )


if __name__ == "__main__":
    logging.getLogger("transformers").setLevel(logging.ERROR)
    audit_modernbert_onnx()
