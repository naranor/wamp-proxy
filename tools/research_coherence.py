import numpy as np
import types
import os
from src.wamp.core.filter import WAMPruner
from benchmarks.coherence_test_long import generate_coherence_long

def run_step_4_coherence():
    print("=== STEP 4: COHERENCE (SUMMARIZATION) RESEARCH ===")
    
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    # 114 messages scenario
    msgs = generate_coherence_long()
    task_query = "Summarize the technical specification of the architecture."
    full_task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{task_query}'"
    critical_indices = [1, 15, 29, 43, 57, 71, 85, 99] # All spec fragments

    algorithms = ["mean_max", "max_max", "mean_mean", "cls_max"]
    multipliers = np.arange(0.90, 1.41, 0.01)
    
    results = {algo: [] for algo in algorithms}

    for algo in algorithms:
        print(f"\nProcessing Algorithm: {algo}")
        
        def custom_batch(self, batch_data, task_ids, task_len, scores_dict):
            input_ids = np.zeros((len(batch_data), task_len + max(i["len"] for i in batch_data)), dtype=np.int64)
            attention_mask = np.zeros_like(input_ids)
            for i, item in enumerate(batch_data):
                seq = task_ids + item["ids"]
                input_ids[i, :len(seq)] = seq
                attention_mask[i, :len(seq)] = 1
            outputs = self.session.run(self.output_names, {'input_ids': input_ids, 'attention_mask': attention_mask})
            for i, item in enumerate(batch_data):
                importance = 0.0
                for layer_attn in outputs:
                    slice = layer_attn[i, :, :task_len, task_len:task_len+item["len"]]
                    if slice.size == 0: continue
                    if algo == "mean_max": importance += float(np.mean(np.max(slice, axis=-1)))
                    elif algo == "max_max": importance += float(np.max(slice))
                    elif algo == "mean_mean": importance += float(np.mean(slice))
                    elif algo == "cls_max": importance += float(np.max(slice[:, 0, :]))
                scores_dict[item["idx"]] = importance / len(self.output_names)

        pruner._process_batch = types.MethodType(custom_batch, pruner)
        _, scores, _, baseline = pruner.get_attention_filtered(msgs, full_task, keep_last_n=2)
        
        for mult in multipliers:
            threshold = baseline * mult
            kept = [i for i, s in enumerate(scores) if i == 0 or i >= len(msgs)-2 or s >= threshold]
            recall = len([idx for idx in critical_indices if idx in kept]) / len(critical_indices) * 100
            savings = (1 - len(kept)/len(msgs)) * 100
            results[algo].append({"mult": mult, "savings": savings, "recall": recall})

    with open("benchmarks/RESEARCH_COHERENCE.md", "w") as f:
        f.write("# Research Step 4: Coherence Audit (114 msgs)\n")
        f.write("**Model:** DeBERTa-v3-small-NLI (INT8)\n\n")
        for algo in algorithms:
            f.write(f"### Algorithm: {algo}\n")
            f.write("| Multiplier | Compression % | Recall (Structure) % | Status |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for res in results[algo]:
                status = "✅ PASS" if res["recall"] == 100 else ("⚠️ PARTIAL" if res["recall"] > 0 else "❌ FAIL")
                f.write(f"| {res['mult']:.2f} | {res['savings']:.1f}% | {res['recall']:.0f}% | {status} |\n")
            f.write("\n")
            
    print("🚀 Research results saved to benchmarks/RESEARCH_COHERENCE.md")

if __name__ == "__main__":
    import logging
    logging.getLogger("wamp").setLevel(logging.ERROR)
    run_step_4_coherence()
