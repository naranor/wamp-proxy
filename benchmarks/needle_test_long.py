import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/v1/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"

def generate_haystack(needle_idx, total_msgs=50):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Always follow the project rules."}
    ]

    noise_topics = [
        "What's the best way to cook salmon?",
        "How do satellites stay in orbit?",
        "I'm thinking about learning photography.",
        "The evolution of jazz music is fascinating.",
        "Why is the sky blue during the day?",
        "Tips for organizing a small workspace.",
        "The history of ancient civilizations.",
        "How to improve your typing speed."
    ]

    for i in range(1, total_msgs + 1):
        if i == needle_idx:
            messages.append({"role": "user", "content": "CRITICAL RULE: Our project uses the 'Argon2id' algorithm for all password hashing."})
            messages.append({"role": "assistant", "content": "Understood. The project exclusively uses Argon2id for password hashing."})
        else:
            topic = noise_topics[i % len(noise_topics)]
            messages.append({"role": "user", "content": f"Topic {i}: {topic}"})
            messages.append({"role": "assistant", "content": f"Regarding '{topic}', it's an interesting subject that many people enjoy exploring in their free time."})

    messages.append({"role": "user", "content": "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name."})
    return messages

async def ask_model(name, url, messages, expected="argon2id", model="auto"):
    print(f"\n--- {name} ---")
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 10}
    
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            elapsed = time.time() - start_time
            choice = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            tokens = data.get("usage", {}).get("prompt_tokens", 0)
            
            is_correct = expected.lower() in choice.lower()
            status = "✅ PASSED" if is_correct else "❌ FAILED"
            print(f"[{status}] Ответ: '{choice}' | Время: {elapsed:.2f}с | Токены: {tokens}")
            return {"tokens": tokens, "correct": is_correct}
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def main():
    print("=== BENCHMARK: Needle in a Haystack (Long: 100+ msgs) ===")
    
    model_name = "auto"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://192.168.92.2:8383/api/v1/models", timeout=5)
            model_name = res.json()["data"][0]["id"]
    except: pass

    messages = generate_haystack(needle_idx=25, total_msgs=55) # ~112 msgs
    print(f"Messages: {len(messages)}")

    direct = await ask_model("DIRECT", DIRECT_URL, messages, model=model_name)
    proxy = await ask_model("WAMP PROXY", PROXY_URL, messages, model=model_name)

    if direct and proxy:
        savings = (1 - (proxy["tokens"] / direct["tokens"])) * 100
        print(f"\n🎉 SAVINGS: {savings:.1f}% | Correctness: {'✅' if proxy['correct'] else '❌'}")

if __name__ == "__main__":
    asyncio.run(main())
