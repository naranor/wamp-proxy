import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/v1/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"

DOCUMENTS = [
    "Project 'Alpha' focuses on quantum computing since 2021.",
    "Dr. Elena Rossi leads Project 'Alpha'.",
    "Project 'Beta' works on sustainable energy since 2022.",
    "Dr. Marcus Chen leads the 'Beta' team.",
    "Alpha budget: $5 million. Beta budget: $3 million.",
    "Alpha uses Python, Beta uses Julia.",
    "Project 'Gamma' was cancelled in 2023.",
    "CyberShield manages security protocols for all projects.",
]


def generate_multi_doc_long():
    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Use the provided fragments to answer questions.",
        }
    ]
    for i, doc in enumerate(DOCUMENTS):
        messages.append({"role": "user", "content": f"Doc {i + 1}: {doc}"})
        messages.append({"role": "assistant", "content": "Fragment received."})
        # Add a lot of noise between documents (6 pairs per doc)
        for j in range(6):
            messages.append(
                {"role": "user", "content": f"Random chat {i}-{j}: Tell me about topic {j}."}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Topic {j} is not relevant to our project data, but interesting.",
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
        }
    )
    return messages


async def run_test(name, url, messages, model="auto"):
    print(f"\n--- {name} ---")
    payload = {"model": model, "messages": messages, "temperature": 0.0}
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            elapsed = time.time() - start_time
            answer = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["prompt_tokens"]
            print(f"⏱ Time: {elapsed:.2f}s | 📊 Tokens: {tokens}")
            print(f"🤖 Answer: {answer.strip()}")
            return tokens
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    print("=== BENCHMARK: Multi-Doc QA (Long: 100+ msgs) ===")
    messages = generate_multi_doc_long()
    print(f"Total messages: {len(messages)}")

    model_name = "auto"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://192.168.92.2:8383/api/v1/models")
            model_name = res.json()["data"][0]["id"]
    except Exception:
        pass

    t_direct = await run_test("DIRECT", DIRECT_URL, messages, model=model_name)
    t_proxy = await run_test("WAMP PROXY", PROXY_URL, messages, model=model_name)

    if t_direct and t_proxy:
        print(f"\n🎉 SAVINGS: {(1 - t_proxy / t_direct) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
