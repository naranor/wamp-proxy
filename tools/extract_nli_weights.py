from transformers import AutoModelForSequenceClassification
import json


def extract_nli():
    model_id = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c"
    print(f"Loading weights from {model_id}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # DeBERTa v3 classification head is usually:
    # pooler -> dropout -> classifier
    # Or just a linear layer on top of CLS

    weights = {
        "classifier_weight": model.classifier.weight.detach().numpy().tolist(),
        "classifier_bias": model.classifier.bias.detach().numpy().tolist(),
        "id2label": model.config.id2label,
    }

    with open("src/wamp/core/nli_weights.json", "w") as f:
        json.dump(weights, f)

    print("🚀 NLI weights saved to src/wamp/core/nli_weights.json")


if __name__ == "__main__":
    extract_nli()
