import numpy as np
import types
import os
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long


def run_max_max_calibration():
    print("=== RESEARCH: MAX-MAX ALGORITHM CALIBRATION (Long Context) ===")

    model_path = "./deberta_small_model"
    if not os.path.exists(model_path):
        print("Error: DeBERTa model not found.")
        return

    pruner = WAMPruner(model_dir=model_path)

    scenarios = {
        "Needle In Haystack": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name.",
            "critical": [51, 52],
        },
        "Multi-Doc QA": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113],
        },
        "Coherence": {
            "msgs": generate_coherence_long(),
            "task": "Provide a comprehensive summary of the architecture. Mention the database, cache, and monitoring tools explicitly.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99],
        },
    }

    # Algorithm: Max-Max
    def custom_batch(self, batch_data, task_ids, task_len, scores_dict):
        batch_size = len(batch_data)
        max_len = task_len + max(item["len"] for item in batch_data)
        input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        for i, item in enumerate(batch_data):
            seq = task_ids + item["ids"]
            input_ids[i, : len(seq)] = seq
            attention_mask[i, : len(seq)] = 1

        outputs = self.session.run(
            self.output_names, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        for i, item in enumerate(batch_data):
            orig_idx = item["idx"]
            msg_len = item["len"]
            importance = 0.0
            for layer_attn in outputs:
                slice = layer_attn[i, :, :task_len, task_len : task_len + msg_len]
                if slice.size > 0:
                    importance += float(np.max(slice))
            scores_dict[orig_idx] = importance / len(self.output_names)

    pruner._process_batch = types.MethodType(custom_batch, pruner)

    multipliers = np.arange(0.90, 1.01, 0.01)

    print(f"{'Mult':<6} | {'Needle (S/R)':<12} | {'QA (S/R)':<12} | {'Coherence (S/R)':<12}")
    print("-" * 50)

    for mult in multipliers:
        line = f"{mult:.2f} | "
        for s_name, s_data in scenarios.items():
            full_task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{s_data['task']}'"
            _, scores, _, baseline = pruner.get_attention_filtered(
                s_data["msgs"], full_task, keep_last_n=2
            )

            threshold = baseline * mult
            kept = [
                i
                for i, s in enumerate(scores)
                if i == 0 or i >= len(s_data["msgs"]) - 2 or s >= threshold
            ]
            recall = (
                len([idx for idx in s_data["critical"] if idx in kept])
                / len(s_data["critical"])
                * 100
            )
            savings = (1 - len(kept) / len(s_data["msgs"])) * 100
            line += f"{savings:>4.1f}%/{recall:>3.0f}% | "
        print(line)


if __name__ == "__main__":
    import logging

    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    run_max_max_calibration()
