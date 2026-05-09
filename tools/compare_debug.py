import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer
from setfit import SetFitModel


def compare_debug():
    tokenizer = AutoTokenizer.from_pretrained("./modernbert_router_model")
    text = "What is the database port?"
    inputs = tokenizer(text, return_tensors="pt")

    # PT
    model_pt = SetFitModel.from_pretrained("./modernbert_router_model")
    with torch.no_grad():
        pt_emb = model_pt.model_body.encode([text])[0]

    # Debug ONNX
    session = ort.InferenceSession("model_debug.onnx")
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
    }
    onnx_outputs = session.run(None, onnx_inputs)
    last_hidden = onnx_outputs[0]

    mean_pooling = np.mean(last_hidden, axis=1)[0]

    print(f"PT (first 5): {pt_emb[:5]}")
    print(f"Debug ONNX Mean (first 5): {mean_pooling[:5]}")
    print(f"Diff: {np.linalg.norm(pt_emb - mean_pooling)}")


if __name__ == "__main__":
    compare_debug()
