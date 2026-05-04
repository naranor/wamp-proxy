import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import json
import os
import logging

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def audit_setfit_onnx():
    print("=== AUDIT: SETFIT ONNX ROUTER (INT4 COMPARISON) ===")
    model_dir = "./model_setfit_onnx"
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 2. Models to test
    models = [
        ("INT8", "model_quantized.onnx"),
        ("INT4", "model_int4.onnx")
    ]
    
    # 3. Load LogReg Head
    with open(os.path.join(model_dir, "router_weights_setfit.json"), "r") as f:
        head = json.load(f)
    coef = np.array(head["coef"])
    intercept = np.array(head["intercept"])
    label_map = {v: k for k, v in head["label_map"].items()}

    test_queries = [
        "What is the database port?",
        "Compare Alpha and Beta projects.",
        "Give me a summary of the chat.",
        "What was the database port mentioned earlier?",
    ]

    for model_name, model_file in models:
        print(f"\n--- Testing {model_name} ({model_file}) ---")
        session = ort.InferenceSession(os.path.join(model_dir, model_file))
        
        print(f"{'Query':<45} | {'Summary':<8} | {'Needle':<8} | {'Reason':<8} | {'Winner'}")
        print("-" * 95)

        for query in test_queries:
            inputs = tokenizer(query, return_tensors="np")
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            outputs = session.run(None, onnx_inputs)
            last_hidden = outputs[0]
            embeddings = np.mean(last_hidden, axis=1)
            scores = np.dot(embeddings, coef.T) + intercept
            exp_scores = np.exp(scores - np.max(scores))
            probs = (exp_scores / exp_scores.sum())[0]
            winner_idx = np.argmax(probs)
            winner_label = label_map[winner_idx]
            print(f"{query[:43]:<45} | {probs[0]:.3f}   | {probs[1]:.3f}   | {probs[2]:.3f}   | {winner_label}")

if __name__ == "__main__":
    logging.getLogger("transformers").setLevel(logging.ERROR)
    audit_setfit_onnx()
