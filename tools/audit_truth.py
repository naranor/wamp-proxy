import numpy as np
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long

def run_token_audit():
    print("=== TRUTHFUL TOKEN-BASED AUDIT (REAL SAVINGS) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    scenarios = {
        "Needle (Fact Retrieval)": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "What was the database port mentioned earlier?",
            "critical": [51, 52]
        },
        "Reasoning (Multi-Doc QA)": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113]
        }
    }

    print(f"{'Scenario':<30} | {'Msg Sav%':<10} | {'Token Sav%':<10} | {'Recall%'}")
    print("-" * 75)

    for name, data in scenarios.items():
        msgs = data["msgs"]
        task_query = data["task"]
        critical = data["critical"]
        
        # Calculate original tokens
        total_tokens_before = sum(len(pruner.tokenizer.encode(m["content"]).ids) for m in msgs)
        
        task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{task_query}'"
        filtered, _, _, _ = pruner.get_attention_filtered(msgs, task, original_query=task_query, keep_last_n=2)
        
        # Calculate filtered tokens
        total_tokens_after = sum(len(pruner.tokenizer.encode(m["content"]).ids) for m in filtered)
        
        kept_indices = [i for i, m in enumerate(msgs) if any(m["content"] == f["content"] for f in filtered)]
        recall = len([idx for idx in critical if idx in kept_indices]) / len(critical) * 100
        
        msg_savings = (1 - len(filtered)/len(msgs)) * 100
        token_savings = (1 - total_tokens_after/total_tokens_before) * 100
        
        print(f"{name:<30} | {msg_savings:>9.1f}% | {token_savings:>9.1f}% | {recall:>7.0f}%")

if __name__ == "__main__":
    import logging
    logging.getLogger("wamp").setLevel(logging.WARNING)
    run_token_audit()
