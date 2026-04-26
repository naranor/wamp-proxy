import numpy as np
import types
from src.wamp.core.filter import WAMPruner
from benchmarks.multi_doc_qa import generate_multi_doc_context

def run_deberta_tuning():
    print("=== DEBERTA-V3-SMALL TUNING (MULTI-DOC QA) ===")
    
    # Load DeBERTa
    pruner = WAMPruner(model_dir="./deberta_small_model")
    
    question = "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?"
    messages = generate_multi_doc_context(question)
    
    # Indices that MUST be kept for the answer
    # Alpha info: [1, 5, 17], Beta info: [9, 13, 17]
    critical_indices = [1, 5, 9, 13, 17] 
    
    prompts = {
        "Simple": question,
        "Instructional": f"Instruction: Extract all facts related to Alpha, Beta, budget, or scientist. Question: {question}",
        "Keywords": "Alpha Beta budget scientist funding lead Dr."
    }
    
    strategies = ["cls_max", "mean_max"]
    
    best_config = None
    max_separation = 0

    for p_name, p_text in prompts.items():
        for strategy in strategies:
            print(f"\n--- Testing: {p_name} | Strategy: {strategy} ---")
            
            # Monkey-patch scoring logic
            def custom_batch(self, batch_data, task_ids, task_len, scores_dict):
                input_ids = np.zeros((len(batch_data), task_len + max(i["len"] for i in batch_data)), dtype=np.int64)
                attention_mask = np.zeros_like(input_ids)
                for i, item in enumerate(batch_data):
                    seq = task_ids + item["ids"]
                    input_ids[i, :len(seq)] = seq
                    attention_mask[i, :len(seq)] = 1
                
                outputs = self.session.run(self.output_names, {'input_ids': input_ids, 'attention_mask': attention_mask})
                
                for i, item in enumerate(batch_data):
                    msg_len = item["len"]
                    importance = 0.0
                    for layer_attn in outputs:
                        if strategy == "cls_max":
                            # Attention from CLS (0) to message
                            slice = layer_attn[i, :, 0, task_len:task_len+msg_len]
                        else:
                            # Attention from ALL task tokens to message
                            slice = layer_attn[i, :, :task_len, task_len:task_len+msg_len]
                        
                        if slice.size > 0:
                            importance += float(np.max(slice))
                    scores_dict[item["idx"]] = importance / len(self.output_names)

            pruner._process_batch = types.MethodType(custom_batch, pruner)
            
            _, scores, _, baseline = pruner.get_attention_filtered(messages, p_text, keep_last_n=2)
            
            critical_scores = [scores[i] for i in critical_indices]
            noise_scores = [scores[i] for i in range(len(messages)) if i not in critical_indices and i != 0 and i < len(messages)-2]
            
            avg_critical = np.mean(critical_scores)
            avg_noise = np.mean(noise_scores)
            separation = avg_critical / (avg_noise + 1e-9)
            
            print(f"Avg Critical: {avg_critical:.6f}")
            print(f"Avg Noise:    {avg_noise:.6f}")
            print(f"Separation:   {separation:.2f}x")
            
            if separation > max_separation:
                max_separation = separation
                best_config = (p_name, strategy, separation)

    print("\n\n=== TUNING RESULT ===")
    print(f"Best Configuration: {best_config[0]} prompt with {best_config[1]} strategy")
    print(f"Max Separation achieved: {best_config[2]:.2f}x")

if __name__ == "__main__":
    import logging
    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    run_deberta_tuning()
