import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/v1/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"


def generate_long_history(total_msgs=100):
    messages = [
        {
            "role": "system",
            "content": "You are a senior DevOps engineer. Help the user with their infrastructure questions.",
        }
    ]

    # Early critical fact (Msg index 5-6)
    messages.append(
        {
            "role": "user",
            "content": "Wait, I forgot to tell you: our production database is running on a non-standard port 9999.",
        }
    )
    messages.append(
        {"role": "assistant", "content": "Got it. Database port is 9999. I'll keep that in mind."}
    )

    # Middle critical fact (Msg index 50-51)
    # We'll fill up to 50
    for i in range(1, 23):
        messages.append(
            {"role": "user", "content": f"Can we discuss the backup strategy for module {i}?"}
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Module {i} backup should be performed daily at midnight using an incremental snapshot.",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": "Also, the backup encryption key is 'WAMP-SECURE-2026'. Never forget this.",
        }
    )
    messages.append(
        {"role": "assistant", "content": "I have noted the encryption key: WAMP-SECURE-2026."}
    )

    # More noise up to 100 with diverse content
    noise_topics = [
        "What's your favorite pizza topping?",
        "Do you think aliens exist in our galaxy?",
        "I'm thinking of buying a new mechanical keyboard.",
        "The weather today is quite unpredictable.",
        "Have you seen the latest superhero movie?",
        "I need a good recipe for chocolate chip cookies.",
        "How many steps should I walk daily for good health?",
        "I love the sound of rain on a tin roof.",
        "Mountain hiking is my favorite weekend activity.",
        "Coffee or tea? That is the eternal question.",
    ]

    for i in range(23, 46):
        topic = noise_topics[i % len(noise_topics)]
        messages.append({"role": "user", "content": f"{topic} (Ref: {i})"})
        messages.append(
            {
                "role": "assistant",
                "content": f"That's a very human topic! Regarding '{topic}', I find it fascinating how varied opinions can be. It certainly adds flavor to life.",
            }
        )

    # Final question
    messages.append(
        {
            "role": "user",
            "content": "I need to restore the production database now. Which port should I connect to and what is the encryption key for the backup? Answer concisely.",
        }
    )

    return messages


async def run_benchmark(name, url, messages, model="auto"):
    print(f"\n--- Running: {name} ---")
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 100}

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            elapsed = time.time() - start_time
            answer = data["choices"][0]["message"]["content"].strip()
            tokens = data["usage"]["prompt_tokens"]

            print(f"⏱ Time: {elapsed:.2f}s | 📊 Prompt Tokens: {tokens}")
            print(f"🤖 Answer: {answer}")
            return tokens
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    print("=== BENCHMARK: 100-Message Long Chat Test ===")
    messages = generate_long_history(total_msgs=100)
    print(f"Total messages generated: {len(messages)}")

    # Auto-detect model
    model_name = "auto"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://192.168.92.2:8383/api/v1/models", timeout=5)
            if res.status_code == 200:
                model_name = res.json()["data"][0]["id"]
    except Exception:
        pass

    t_direct = await run_benchmark(
        "DIRECT (No Compression)", DIRECT_URL, messages, model=model_name
    )
    t_proxy = await run_benchmark(
        "WAMP PROXY (Smart Compression)", PROXY_URL, messages, model=model_name
    )

    if t_direct and t_proxy:
        savings = (1 - t_proxy / t_direct) * 100
        print("\n==================================================")
        print(f"🎉 FINAL SAVINGS ON 100 MSGS: {savings:.1f}%")
        print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
