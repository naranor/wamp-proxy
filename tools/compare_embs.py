from setfit import SetFitModel
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer


def compare_embeddings():
    model_pt = SetFitModel.from_pretrained("./modernbert_router_model")
    tokenizer = AutoTokenizer.from_pretrained("./modernbert_router_model")

    text = "What is the database port?"

    # 1. PT Embedding
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        pt_emb = model_pt.model_body.encode([text])[0]

    # 2. ONNX Embedding
    session = ort.InferenceSession("./model_modernbert_onnx/model_quantized.onnx")
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
    }
    onnx_outputs = session.run(None, onnx_inputs)
    last_hidden = onnx_outputs[0]

    # Try different poolings
    mean_pooling = np.mean(last_hidden, axis=1)[0]
    cls_pooling = last_hidden[0, 0, :]

    print(f"PT Embedding (first 5): {pt_emb[:5]}")
    print(f"ONNX Mean Pooling (first 5): {mean_pooling[:5]}")
    print(f"ONNX CLS Pooling (first 5): {cls_pooling[:5]}")

    # Check similarity
    print(f"Diff Mean: {np.linalg.norm(pt_emb - mean_pooling)}")
    print(f"Diff CLS: {np.linalg.norm(pt_emb - cls_pooling)}")


if __name__ == "__main__":
    compare_embeddings()
