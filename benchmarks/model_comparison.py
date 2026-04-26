import numpy as np
import os
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test import generate_haystack
import src.wamp.core.config as config

def run_comparative_benchmark():
    print("=== COMPARATIVE ATTENTION AUDIT ===")
    
    models = {
        "ModernBERT-base": "modernbert_model",
        "DeBERTa-v3-small": "deberta_small_model",
        "ModernBERT-large": "modernbert_large_model"
    }
    
    # Generate common test case
    messages = generate_haystack(needle_idx=15, total_msgs=30)
    user_query = "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name."
    task = user_query
    needle_indices = [29, 30]
    
    results = {}

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (path not found)")
            continue
            
        print(f"\n--- Testing Model: {name} ---")
        
        try:
            # Explicitly pass model_dir to constructor
            pruner = WAMPruner(model_dir=path)
            _, scores, threshold, baseline = pruner.get_attention_filtered(messages, task, keep_last_n=2)
            
            needle_scores = [scores[i] for i in needle_indices]
            noise_scores = [scores[i] for i in range(len(messages)) if i not in needle_indices and i != 0 and i < len(messages)-2]
            
            mean_needle = np.mean(needle_scores)
            max_noise = np.max(noise_scores)
            sep_ratio = mean_needle / (max_noise + 1e-9)
            
            results[name] = {
                "baseline": baseline,
                "needle_avg": mean_needle,
                "noise_max": max_noise,
                "separation": sep_ratio,
                "passed": mean_needle > max_noise
            }
            
            print(f"Baseline (Median): {baseline:.6f}")
            print(f"Needle Avg Score: {mean_needle:.6f}")
            print(f"Noise Max Score:  {max_noise:.6f}")
            print(f"Separation Ratio: {sep_ratio:.2f}x")
            print(f"Result: {'✅ PASSED' if mean_needle > max_noise else '❌ FAILED'}")
            
        except Exception as e:
            print(f"Error testing {name}: {e}")

    print("\n\n=== FINAL COMPARISON SUMMARY ===")
    print(f"{'Model':<20} | {'Separation':<12} | {'Status'}")
    print("-" * 45)
    for name, data in results.items():
        status = "✅" if data["passed"] else "❌"
        print(f"{name:<20} | {data['separation']:<12.2f} | {status}")

if __name__ == "__main__":
    import logging
    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    run_comparative_benchmark()
