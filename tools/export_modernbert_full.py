import torch
from setfit import SetFitModel
from transformers import AutoTokenizer
import os
import json


def export_modernbert_complete():
    print("=== EXPORTING TRAINED ModernBERT TO ONNX (LAST 2 LAYERS) ===")
    src_dir = "./modernbert_router_model"
    dest_dir = "./model_modernbert_onnx"
    os.makedirs(dest_dir, exist_ok=True)

    # 1. Load trained model
    model = SetFitModel.from_pretrained(src_dir)
    transformer = model.model_body[0].auto_model
    tokenizer = AutoTokenizer.from_pretrained(src_dir)

    # 2. Wrap for attentions (Optimized: only last 2 layers)
    class ModernBertWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
            )
            # ModernBERT-base has 22 layers. We return last_hidden + 2 attention matrices.
            return outputs.last_hidden_state, outputs.attentions[-2], outputs.attentions[-1]

    wrapper = ModernBertWrapper(transformer)
    wrapper.eval()

    # 3. Export
    dummy_text = "test"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    num_layers = len(transformer.layers)
    input_names = ["input_ids", "attention_mask"]
    output_names = [
        "last_hidden_state",
        f"attentions.{num_layers - 2}",
        f"attentions.{num_layers - 1}",
    ]

    onnx_path = os.path.join(dest_dir, "model.onnx")
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        wrapper,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=18,
    )

    # 4. Save Router Weights
    head = model.model_head
    weights = {
        "coef": head.coef_.tolist(),
        "intercept": head.intercept_.tolist(),
        "label_map": {"Summary": 0, "Needle": 1, "Reasoning": 2},
    }
    with open(os.path.join(dest_dir, "router_weights_setfit.json"), "w") as f:
        json.dump(weights, f)

    # 5. Copy Tokenizer
    tokenizer.save_pretrained(dest_dir)
    print(f"✅ Optimized ModernBERT package ready in {dest_dir}")


if __name__ == "__main__":
    export_modernbert_complete()
