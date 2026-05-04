from setfit import SetFitModel, SetFitTrainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import os
import logging

def load_dataset_custom(filename):
    """Load DATA list from a python file manually."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        loc = {}
        exec(content, {}, loc)
        return loc.get("DATA", [])

def train_setfit():
    print("=== TRAINING SETFIT SEMANTIC ROUTER ===")
    
    # 1. Load raw data
    summary_tasks = load_dataset_custom("tools/dataset_task_summary.py")
    needle_tasks = load_dataset_custom("tools/dataset_task_needle.py")
    reasoning_tasks = load_dataset_custom("tools/dataset_task_reasoning.py")
    
    # 2. Convert to HuggingFace Dataset format
    # Labels: 0: Summary, 1: Needle, 2: Reasoning
    texts = summary_tasks + needle_tasks + reasoning_tasks
    labels = [0]*len(summary_tasks) + [1]*len(needle_tasks) + [2]*len(reasoning_tasks)
    
    df = pd.DataFrame({"text": texts, "label": labels})
    dataset = Dataset.from_pandas(df)
    
    # Shuffle
    dataset = dataset.shuffle(seed=42)

    # 3. Load Base Model
    # Multilingual MiniLM is small (~100MB) and very capable
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 5. Trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset,
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
        column_mapping={"text": "text", "label": "label"}
    )

    # 6. Run Training
    print("Starting Contrastive Fine-tuning...")
    trainer.train()
    
    # 7. Save the model
    save_path = "./setfit_router_model"
    model.save_pretrained(save_path)
    print(f"🚀 SetFit Router model saved to {save_path}")

if __name__ == "__main__":
    # Suppress warnings
    logging.getLogger("setfit").setLevel(logging.ERROR)
    train_setfit()
