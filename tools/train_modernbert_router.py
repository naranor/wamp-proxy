from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import pandas as pd
import logging


def load_dataset_custom(filename):
    """Load DATA list from a python file manually."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        loc = {}
        exec(content, {}, loc)
        return loc.get("DATA", [])


def train_modernbert_setfit():
    print("=== TRAINING ModernBERT-BASE SETFIT ROUTER ===")

    # 1. Load raw data
    summary_tasks = load_dataset_custom("tools/dataset_task_summary.py")
    needle_tasks = load_dataset_custom("tools/dataset_task_needle.py")
    reasoning_tasks = load_dataset_custom("tools/dataset_task_reasoning.py")

    # 2. Convert to Dataset format
    texts = summary_tasks + needle_tasks + reasoning_tasks
    labels = [0] * len(summary_tasks) + [1] * len(needle_tasks) + [2] * len(reasoning_tasks)

    df = pd.DataFrame({"text": texts, "label": labels})
    dataset = Dataset.from_pandas(df).shuffle(seed=42)

    # 3. Load ModernBERT Model
    # We use the base model for training to get compatible embeddings
    print("Loading answerdotai/ModernBERT-base...")
    model = SetFitModel.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)

    # 4. Training
    print("Starting training (this may take longer than MiniLM)...")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset,
        batch_size=8,  # ModernBERT is larger, use smaller batch
        num_epochs=1,
        num_iterations=20,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()

    # 5. Save results
    save_path = "./modernbert_router_model"
    model.save_pretrained(save_path)
    print(f"🚀 ModernBERT Router model saved to {save_path}")


if __name__ == "__main__":
    logging.getLogger("setfit").setLevel(logging.ERROR)
    train_modernbert_setfit()
