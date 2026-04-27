import numpy as np
from src.wamp.core.filter import WAMPruner
import json

def reconstruct_from_mask(tokens, mask, tokenizer):
    """Reconstruct message text using only tokens in the mask."""
    result = []
    in_gap = False
    
    for i, token_id in enumerate(tokens):
        if mask[i]:
            if in_gap:
                result.append("...")
                in_gap = False
            result.append(tokenizer.decode([token_id]))
        else:
            in_gap = True
            
    return " ".join(result).replace(" . . . ", "...").strip()

def run_token_pruning_experiment():
    print("=== RESEARCH STEP 5: TOKEN-LEVEL PRUNING ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    # Test message (High water, low facts)
    msg = "I am happy to inform you that our project Alpha, which started back in 2021, is currently allocated a budget of 5 million dollars. Dr. Elena Rossi is leading it."
    task = "What is the budget of project Alpha?"
    
    print(f"\nOriginal Message ({len(pruner.tokenizer.encode(msg).ids)} tokens):")
    print(msg)
    
    # Get attentions
    task_enc = pruner.tokenizer.encode(f"TASK: {task}\n\n")
    msg_enc = pruner.tokenizer.encode(f"Msg: {msg}\n")
    
    input_ids = np.array([task_enc.ids + msg_enc.ids], dtype=np.int64)
    mask = np.array([[1] * len(input_ids[0])], dtype=np.int64)
    
    outputs = pruner.session.run(pruner.output_names, {'input_ids': input_ids, 'attention_mask': mask})
    
    # Use last layer attention
    # shape: (batch, heads, task_len, msg_len)
    last_attn = outputs[-1][0]
    task_len = len(task_enc.ids)
    msg_len = len(msg_enc.ids)
    
    # Cross-attention slice from Task to Msg
    slc = last_attn[:, :task_len, task_len:] # (heads, q, m)
    
    # Max attention for each message token across all heads and all task tokens
    token_importance = np.max(np.max(slc, axis=0), axis=0)
    
    print("\n--- Testing Thresholds ---")
    for threshold in [0.01, 0.05, 0.1, 0.2]:
        token_mask = token_importance > threshold
        reconstructed = reconstruct_from_mask(msg_enc.ids, token_mask, pruner.tokenizer)
        savings = (1 - sum(token_mask)/msg_len) * 100
        print(f"Threshold {threshold:.2f} | Savings {savings:>4.1f}% | Result: {reconstructed}")

if __name__ == "__main__":
    import logging
    logging.getLogger("wamp").setLevel(logging.ERROR)
    run_token_pruning_experiment()
