import numpy as np
import types
import os
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long

def run_algo_comparison():
    print("=== RESEARCH: ATTENTION AGGREGATION ALGORITHMS ===")
    
    # Use DeBERTa as the primary research model
    model_path = "./deberta_small_model"
    if not os.path.exists(model_path):
        print("Error: DeBERTa model not found. Please export it first.")
        return
        
    pruner = WAMPruner(model_dir=model_path)
    
    scenarios = {
        "Needle In Haystack": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name.",
            "critical": [51, 52]
        },
        "Multi-Doc QA": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113]
        },
        "Coherence": {
            "msgs": generate_coherence_long(),
            "task": "Provide a comprehensive summary of the architecture. Mention the database, cache, and monitoring tools explicitly.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99]
        }
    }

    # Define algorithms to test
    # 1. Mean-MAX (Current): Mean of (Max attention from each task token to msg)
    # 2. CLS-MAX: Max attention from CLS token only to msg
    # 3. Mean-Mean: Mean attention from all task tokens to all msg tokens
    # 4. Max-Max: Absolute max attention in the task-msg cross-matrix
    
    algorithms = ["mean_max", "cls_max", "mean_mean", "max_max"]
    multiplier = 0.98
    
    results = {}

    for algo in algorithms:
        print(f"\n>>> Testing Algorithm: {algo}")
        
        # Monkey-patch the scoring logic
        def custom_batch(self, batch_data, task_ids, task_len, scores_dict):
            batch_size = len(batch_data)
            max_len = task_len + max(item["len"] for item in batch_data)
            input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
            attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
            for i, item in enumerate(batch_data):
                seq = task_ids + item["ids"]
                input_ids[i, :len(seq)] = seq
                attention_mask[i, :len(seq)] = 1
                
            outputs = self.session.run(self.output_names, {'input_ids': input_ids, 'attention_mask': attention_mask})
            
            for i, item in enumerate(batch_data):
                orig_idx = item["idx"]
                msg_len = item["len"]
                importance = 0.0
                
                for layer_attn in outputs:
                    # slice shape: (heads, task_tokens, msg_tokens)
                    slice = layer_attn[i, :, :task_len, task_len:task_len+msg_len]
                    if slice.size == 0: continue
                    
                    if algo == "mean_max":
                        importance += float(np.mean(np.max(slice, axis=-1)))
                    elif algo == "cls_max":
                        importance += float(np.max(slice[:, 0, :]))
                    elif algo == "mean_mean":
                        importance += float(np.mean(slice))
                    elif algo == "max_max":
                        importance += float(np.max(slice))
                
                scores_dict[orig_idx] = importance / len(self.output_names)

        pruner._process_batch = types.MethodType(custom_batch, pruner)
        
        algo_res = {}
        for s_name, s_data in scenarios.items():
            full_task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{s_data['task']}'"
            _, scores, _, baseline = pruner.get_attention_filtered(s_data['msgs'], full_task, keep_last_n=2)
            
            threshold = baseline * multiplier
            kept = [i for i, s in enumerate(scores) if i == 0 or i >= len(s_data['msgs'])-2 or s >= threshold]
            recall = len([idx for idx in s_data['critical'] if idx in kept]) / len(s_data['critical']) * 100
            savings = (1 - len(kept)/len(s_data['msgs'])) * 100
            algo_res[s_name] = {"savings": savings, "recall": recall}
            
        results[algo] = algo_res

    print("\n\n=== FINAL ALGORITHM COMPARISON (Multiplier 0.98) ===")
    header = f"{'Algorithm':<12} | {'Needle (S/R)':<15} | {'QA (S/R)':<15} | {'Coherence (S/R)':<15}"
    print(header)
    print("-" * len(header))
    
    for algo, res in results.items():
        n = res["Needle In Haystack"]
        q = res["Multi-Doc QA"]
        c = res["Coherence"]
        print(f"{algo:<12} | {n['savings']:>4.1f}%/{n['recall']:>3.0f}% | {q['savings']:>4.1f}%/{q['recall']:>3.0f}% | {c['savings']:>4.1f}%/{c['recall']:>3.0f}%")

if __name__ == "__main__":
    import logging
    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    run_algo_comparison()
