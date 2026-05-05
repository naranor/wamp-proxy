import torch
from setfit import SetFitModel
from transformers import AutoTokenizer
import os
import json


def export_setfit_to_onnx():
    print("=== EXPORTING SETFIT MODEL TO ONNX (WITH ATTENTIONS) ===")
    model_path = "./setfit_router_model"
    export_path = "./model_setfit_onnx"
    os.makedirs(export_path, exist_ok=True)

    # 1. Load SetFit Model
    model = SetFitModel.from_pretrained(model_path)
    # The transformer body is in model.model_body
    transformer = model.model_body[0].auto_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Prepare dummy input for ONNX export
    dummy_text = "What is the port?"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    # 3. Export Transformer Body
    print("Exporting Transformer body...")
    onnx_body_path = os.path.join(export_path, "model.onnx")

    # We must wrap the model to ensure attentions are returned
    class AttentionWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
            )
            # Return last_hidden_state, pooler_output (if exists), and all attentions
            return outputs.last_hidden_state, *outputs.attentions

    wrapper = AttentionWrapper(transformer)
    wrapper.eval()

    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"] + [
        f"attentions.{i}" for i in range(len(transformer.encoder.layer))
    ]

    torch.onnx.export(
        wrapper,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_body_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    # 4. Save Classifier Head weights (LogisticRegression)
    print("Saving Classifier Head weights...")
    head = model.model_head
    head_weights = {
        "coef": head.coef_.tolist(),
        "intercept": head.intercept_.tolist(),
        "label_map": {"Summary": 0, "Needle": 1, "Reasoning": 2},
    }
    with open(os.path.join(export_path, "router_weights_setfit.json"), "w") as f:
        json.dump(head_weights, f)

    # 5. Copy Tokenizer
    tokenizer.save_pretrained(export_path)

    print(f"\n🚀 Export complete! Files saved to {export_path}")
    print(f"ONNX Model includes {len(transformer.encoder.layer)} attention layers.")


if __name__ == "__main__":
    export_setfit_to_onnx()
