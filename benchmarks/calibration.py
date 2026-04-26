from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test import generate_haystack


def run_final_test():
    print("=== ФИНАЛЬНЫЙ ТЕСТ: MAX ATTENTION + BATCHING ===")
    pruner = WAMPruner()

    # Needle in the middle (classic test)
    messages = generate_haystack(needle_idx=15, total_msgs=30)
    # Improved task prompt (same as in proxy.py)
    user_query = "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name."
    task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{user_query}'"

    needle_indices = [29, 30]

    print(f"Контекст: {len(messages)} сообщений. Иголки: {needle_indices}")

    _, scores, _, mean_score = pruner.get_attention_filtered(messages, task, keep_last_n=2)

    print(f"Средний пиковый балл (Mean): {mean_score:.6f}")

    for mult in [0.8, 0.85, 0.9, 0.95, 1.0]:
        threshold = mean_score * mult
        kept = [
            i for i, s in enumerate(scores) if i == 0 or i >= len(messages) - 2 or s >= threshold
        ]
        needle_saved = all(idx in kept for idx in needle_indices)
        savings = (1 - (len(kept) / len(messages))) * 100

        status = "✅ OK" if needle_saved else "❌ LOST"
        print(
            f"Множитель {mult}: Сжатие {savings:.1f}% | Иголка {status} | Оставлено {len(kept)} сообщений"
        )


if __name__ == "__main__":
    import logging

    logging.getLogger("src.wamp.core.filter").setLevel(logging.WARNING)
    run_final_test()
