from src.wamp.core.filter import WAMPruner
from benchmarks.multi_doc_qa import generate_multi_doc_context


def calibrate_multi_doc():
    print("=== КАЛИБРОВКА ПО MULTI-DOC QA ===")
    pruner = WAMPruner()

    question = (
        "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?"
    )
    messages = generate_multi_doc_context(question)
    # Use instructional prompt for calibration
    task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{question}'"

    # Релевантные индексы (те, где есть инфа про Alpha и Beta)
    # По структуре DOCUMENTS: 1, 2 (Alpha), 3, 4 (Beta), 5 (Budgets), 6 (Simulations)
    # В generate_multi_doc_context на каждый документ 4 сообщения (user, assistant, noise_user, noise_assistant)
    # Плюс 1 системное.
    # Фрагменты Alpha: 1, 2, 5, 6
    # Индексы в messages:
    # 0: system
    # 1: Alpha 1 (User)
    # 5: Alpha 2 (User)
    # 17: Budgets (User)
    # 9: Beta 1 (User)
    # 13: Beta 2 (User)

    needle_indices = []
    for i, m in enumerate(messages):
        content = m.get("content", "")
        if any(x in content for x in ["Alpha", "Beta", "budget", "scientist"]):
            if m.get("role") == "user":  # Только сообщения пользователя содержат инфу
                needle_indices.append(i)

    print(f"Контекст: {len(messages)} сообщений. Релевантные индексы: {needle_indices}")

    _, scores, _, mean_score = pruner.get_attention_filtered(messages, task, keep_last_n=2)

    print(f"Средний пиковый балл (Mean): {mean_score:.6f}")

    # Increase range to find where compression starts
    import numpy as np

    for mult in np.arange(0.9, 1.11, 0.01):
        threshold = mean_score * mult
        kept = [
            i for i, s in enumerate(scores) if i == 0 or i >= len(messages) - 2 or s >= threshold
        ]

        saved_count = sum(1 for idx in needle_indices if idx in kept)
        savings = (1 - (len(kept) / len(messages))) * 100

        status = (
            f"✅ {saved_count}/{len(needle_indices)} SAVED"
            if saved_count == len(needle_indices)
            else f"⚠️ {saved_count}/{len(needle_indices)} saved"
        )
        if saved_count == 0:
            status = "❌ FAILED"

        print(
            f"Множитель {mult}: Сжатие {savings:.1f}% | {status} | Оставлено {len(kept)} сообщений"
        )


if __name__ == "__main__":
    import logging

    logging.getLogger("src.wamp.core.filter").setLevel(logging.WARNING)
    calibrate_multi_doc()
