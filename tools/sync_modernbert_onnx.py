import joblib
import json
import os
from pathlib import Path
import shutil


def sync_modernbert_weights():
    print("=== SYNCING ModernBERT SETFIT WEIGHTS TO ONNX PACKAGE ===")

    src_dir = Path("./modernbert_router_model")
    dest_dir = Path("./model_modernbert_onnx")
    os.makedirs(dest_dir, exist_ok=True)

    # 1. Load the scikit-learn head
    head_path = src_dir / "model_head.pkl"
    if not head_path.exists():
        print(f"Error: {head_path} not found.")
        return

    print(f"Loading weights from {head_path}...")
    clf = joblib.load(head_path)

    # 2. Extract coefficients and intercept
    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "label_map": {"Summary": 0, "Needle": 1, "Reasoning": 2},
        "model_info": "ModernBERT-base SetFit Router",
        "features": "Mean Pooling (768d)",
    }

    # 3. Save to the standardized JSON format
    weights_path = dest_dir / "router_weights_setfit.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"✅ Weights saved to {weights_path}")

    # 4. Copy Tokenizer files (Critical for consistency)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ]
    for f in tokenizer_files:
        if (src_dir / f).exists():
            shutil.copy(src_dir / f, dest_dir / f)
            print(f"✓ Copied {f}")

    print(f"\n🚀 Package ready in {dest_dir}")
    print(
        "Next step: Download 'model.onnx' from naranor/ModernBERT-base-ONNX-Attentions into this folder."
    )


if __name__ == "__main__":
    sync_modernbert_weights()
