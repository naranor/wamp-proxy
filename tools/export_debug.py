import torch
from setfit import SetFitModel


def export_trained_body():
    model = SetFitModel.from_pretrained("./modernbert_router_model")
    # model.model_body is a SentenceTransformer
    # The actual transformer is inside
    transformer = model.model_body[0].auto_model

    dummy_text = "test"
    tokenizer = model.model_body.tokenizer
    inputs = tokenizer(dummy_text, return_tensors="pt")

    torch.onnx.export(
        transformer,
        (inputs["input_ids"], inputs["attention_mask"]),
        "model_debug.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=14,
    )
    print("Exported model_debug.onnx")


if __name__ == "__main__":
    export_trained_body()
