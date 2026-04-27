import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/v1/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"

# База знаний из разрозненных фрагментов
DOCUMENTS = [
    "Project 'Alpha' started in 2021 and focuses on quantum computing.",
    "The lead scientist for Project 'Alpha' is Dr. Elena Rossi.",
    "Project 'Beta' is dedicated to sustainable energy and began in 2022.",
    "Dr. Marcus Chen leads the 'Beta' team in Singapore.",
    "The budget for Alpha is 5 million dollars, while Beta has 3 million.",
    "Alpha uses Python for simulations, and Beta uses Julia.",
    "A third project, 'Gamma', was cancelled in 2023 due to lack of interest.",
    "Security protocols for all projects are managed by the 'CyberShield' department.",
]


def generate_multi_doc_context(question):
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Use the provided project fragments to answer questions.",
        }
    ]

    # Добавляем "шум" между документами
    for i, doc in enumerate(DOCUMENTS):
        messages.append({"role": "user", "content": f"Information fragment {i + 1}: {doc}"})
        messages.append({"role": "assistant", "content": "Received and indexed."})
        # Вставляем совсем нерелевантный шум
        messages.append(
            {
                "role": "user",
                "content": f"Did you know that the sky is blue and cats like module {i}?",
            }
        )
        messages.append(
            {"role": "assistant", "content": "That is an interesting but irrelevant fact."}
        )

    messages.append({"role": "user", "content": question})
    return messages


async def run_qa_test(name, url, messages, model="auto"):
    print(f"\n--- {name} ---")
    payload = {"model": model, "messages": messages, "temperature": 0.0}

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            elapsed = time.time() - start_time
            answer = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["prompt_tokens"]
            print(f"⏱ Время: {elapsed:.2f}с | 📊 Токены: {tokens}")
            print(f"🤖 Ответ: {answer.strip()}")
            return tokens
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


async def main():
    print("=== BENCHMARK: Multi-Document Reasoning ===")
    question = (
        "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?"
    )
    messages = generate_multi_doc_context(question)

    # Auto-detect model
    model_name = "auto"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://localhost:3000/health", timeout=5)
            # Health gives upstream, let's try to get model from direct
            res = await client.get("http://192.168.92.2:8383/api/v1/models")
            model_name = res.json()["data"][0]["id"]
    except Exception:
        pass

    print(f"Общее кол-во сообщений: {len(messages)}")

    t_direct = await run_qa_test(
        "БЕЗ ПРОКСИ (Full Context)", DIRECT_URL, messages, model=model_name
    )
    t_proxy = await run_qa_test("ЧЕРЕЗ ПРОКСИ (Compressed)", PROXY_URL, messages, model=model_name)

    if t_direct and t_proxy:
        print(f"\n🎉 ЭКОНОМИЯ: {(1 - t_proxy / t_direct) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
