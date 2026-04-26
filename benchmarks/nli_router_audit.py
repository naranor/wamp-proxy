import numpy as np
from src.wamp.core.filter import WAMPruner

def test_nli_router():
    print("=== RESEARCH: NLI-BASED ZERO-SHOT ROUTING AUDIT ===")
    # Load the new NLI model
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    test_queries = {
        "Needle": [
            "What is the database port mentioned earlier?",
            "Show me the encryption key.",
            "What is the backup secret?"
        ],
        "Reasoning": [
            "Compare projects Alpha and Beta in terms of budget.",
            "Analyze the differences between the current and future pipelines.",
            "Which team has more funding according to the discussion?"
        ],
        "Summary": [
            "Give me a brief summary of our architecture discussion.",
            "Provide a comprehensive overview of all fragments.",
            "TL;DR of the whole chat history."
        ]
    }

    print(f"{'Category':<12} | {'Predicted':<10} | {'Status'}")
    print("-" * 40)

    correct = 0
    total = 0
    for q_name, queries in test_queries.items():
        for query in queries:
            predicted = pruner.classify_task(query)
            is_correct = (q_name == predicted)
            if is_correct: correct += 1
            total += 1
            print(f"{q_name:<12} | {predicted:<10} | {'✅' if is_correct else '❌'}")

    print(f"\nRouter Accuracy: {correct/total*100:.1f}%")

if __name__ == "__main__":
    import logging
    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    test_nli_router()
