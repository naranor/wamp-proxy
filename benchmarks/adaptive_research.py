import numpy as np
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long

def run_final_adaptive_research():
    print("=== FINAL RESEARCH: HYBRID ADAPTIVE ROUTING (No Proxy) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    scenarios = {
        "Needle (mean_max)": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "What was the database port mentioned earlier?",
            "critical": [51, 52]
        },
        "Reasoning (max_max)": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113]
        },
        "Summary (mean_mean)": {
            "msgs": generate_coherence_long(),
            "task": "Summarize the technical specification of the architecture.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99]
        }
    }

    print(f"{'Scenario':<25} | {'Category':<10} | {'Algo':<10} | {'Savings':<10} | {'Recall'}")
    print("-" * 75)

    for name, data in scenarios.items():
        msgs = data["msgs"]
        task = data["task"]
        critical = data["critical"]
        
        # This will use the live logic from WAMPruner
        filtered, scores, threshold, baseline = pruner.get_attention_filtered(msgs, task, keep_last_n=2)
        
        # Extract classification info from logs isn't easy here, so we look at the results
        # We know what the pruner SHOULD choose
        category = pruner.classify_task(task)
        
        kept_indices = [i for i, m in enumerate(msgs) if any(m["content"] == f["content"] for f in filtered)]
        recall = len([idx for idx in critical if idx in kept_indices]) / len(critical) * 100
        savings = (1 - len(filtered)/len(msgs)) * 100
        
        # Map category to algo name for display
        algo = "mean_max" if category == "Needle" else ("mean_mean" if category == "Summary" else "max_max")
        
        print(f"{name:<25} | {category:<10} | {algo:<10} | {savings:>8.1f}% | {recall:>5.0f}%")

if __name__ == "__main__":
    import logging
    # Set to WARNING to keep output clean but show our result
    logging.getLogger("wamp").setLevel(logging.WARNING)
    run_final_adaptive_research()
