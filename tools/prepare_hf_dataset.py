import pandas as pd
import os

def load_dataset_custom(filename):
    """Load DATA list from a python file manually."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        loc = {}
        exec(content, {}, loc)
        return loc.get("DATA", [])

def prepare_hf_dataset():
    print("=== PREPARING WAMP ROUTER DATASET FOR HUGGING FACE ===")
    
    # 1. Load data
    summary_tasks = load_dataset_custom("tools/dataset_task_summary.py")
    needle_tasks = load_dataset_custom("tools/dataset_task_needle.py")
    reasoning_tasks = load_dataset_custom("tools/dataset_task_reasoning.py")
    
    # 2. Create DataFrame
    # Labels: 0: Summary, 1: Needle, 2: Reasoning
    data = []
    for t in summary_tasks: data.append({"text": t, "label": 0, "label_text": "Summary"})
    for t in needle_tasks: data.append({"text": t, "label": 1, "label_text": "Needle"})
    for t in reasoning_tasks: data.append({"text": t, "label": 2, "label_text": "Reasoning"})
    
    df = pd.DataFrame(data)
    
    # 3. Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 4. Create directory and save
    os.makedirs("wamp_router_dataset", exist_ok=True)
    csv_path = "wamp_router_dataset/dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"✅ Dataset consolidated: {len(df)} samples.")
    print(f"🚀 CSV saved to {csv_path}")

    # 5. Create Dataset Card (README.md)
    readme_content = """---
language:
- ru
- en
license: mit
task_categories:
- text-classification
tags:
- intent-classification
- llm-proxy
- context-compression
- multilingual
size_categories:
- n<1K
---

# WAMP Router Intent Dataset

This dataset was created for training the **[WAMP-proxy](https://github.com/naranor/wamp-proxy)** semantic router. It contains user queries in **Russian** and **English** across various domains (Technical, Medicine, Art, Philosophy, etc.) classified into three intent categories.

## Dataset Structure

The dataset consists of two columns:
- `text`: The user query string.
- `label`: The integer class ID.
- `label_text`: Human-readable class name.

### Labels
- **0 (Summary):** General requests for recaps, TL;DR, or broad overviews of the conversation.
- **1 (Needle):** Specific fact retrieval or parameter extraction (keys, ports, dates, names).
- **2 (Reasoning):** Complex analytical requests involving logic, comparison, debugging, or synthesis.

## Use Case

Ideal for training lightweight intent classifiers for LLM middleware, proxies, or local AI assistants that need to optimize context usage based on task type.

## Project Origin

Part of the **[Weighted Attention Message Pruner](https://github.com/naranor/wamp-proxy)** research project.
"""
    with open("wamp_router_dataset/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ Dataset Card (README.md) created.")

if __name__ == "__main__":
    prepare_hf_dataset()
