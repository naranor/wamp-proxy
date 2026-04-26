import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/v1/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"


def generate_haystack(needle_idx, total_msgs=40):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Always follow the project rules.",
        }
    ]

    for i in range(1, total_msgs + 1):
        if i == needle_idx:
            messages.append(
                {
                    "role": "user",
                    "content": "CRITICAL RULE: Our project uses the 'Argon2id' algorithm for all password hashing.",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood. The project exclusively uses Argon2id for password hashing.",
                }
            )
        else:
            messages.append(
                {"role": "user", "content": f"Discuss the importance of unit testing module {i}."}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Unit testing module {i} is crucial for ensuring software reliability and catching bugs early.",
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name.",
        }
    )
    return messages


async def ask_model(name, url, messages, expected="argon2id", model="auto"):
    print(f"\n--- {name} ---")
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 10}

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                print(f"❌ Error: Upstream returned {response.status_code}")
                print(f"Response body: {response.text}")
                return None

            data = response.json()

            elapsed = time.time() - start_time
            message_obj = data.get("choices", [{}])[0].get("message")
            if not message_obj:
                print(f"❌ Error: Invalid response structure: {data}")
                return None

            choice = message_obj.get("content", "")
            if choice is None:
                choice = ""

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)

            is_correct = expected.lower() in choice.lower()
            status = "✅ PASSED" if is_correct else "❌ FAILED"

            print(f"[{status}] Ответ: '{choice.strip()}' | Время: {elapsed:.2f}с | Токены (входящие): {prompt_tokens}")
            return {"tokens": prompt_tokens, "correct": is_correct, "time": elapsed}

    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
        return None


async def main():
    print("=== CCPROXY BENCHMARK: Needle in a Haystack (Short) ===")

    # Auto-detect model
    model_name = "auto"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://192.168.92.2:8383/api/v1/models", timeout=5)
            if res.status_code == 200 and res.json().get("data"):
                model_name = res.json()["data"][0]["id"]
    except Exception:
        pass

    # Run test
    needle_pos = 15
    total = 30
    messages = generate_haystack(needle_idx=needle_pos, total_msgs=total)

    print(f"\nГенерация контекста: {len(messages)} сообщений. 'Иголка' на позиции {needle_pos}.")

    direct = await ask_model("БЕЗ ПРОКСИ (Прямой вызов)", DIRECT_URL, messages, model=model_name)
    proxy = await ask_model(
        "ЧЕРЕЗ ПРОКСИ (Сжатие ModernBERT)", PROXY_URL, messages, model=model_name
    )

    if direct and proxy:
        savings = (1 - (proxy["tokens"] / direct["tokens"])) * 100
        print("\n==================================================")
        print("📊 РЕЗУЛЬТАТЫ:")
        print(f"Токенов без сжатия: {direct['tokens']}")
        print(f"Токенов со сжатием: {proxy['tokens']}")
        print(f"🎉 ЭКОНОМИЯ КОНТЕКСТА: {savings:.1f}%")
        print(f"🎯 Точность без прокси: {'✅' if direct['correct'] else '❌'}")
        print(f"🎯 Точность с прокси:   {'✅' if proxy['correct'] else '❌'}")
        print("==================================================")

        if not proxy["correct"]:
            print("\n⚠️ ВНИМАНИЕ: Прокси сэкономил токены, но потерял критический контекст!")
            print(
                "Рекомендуется снизить FILTER_THRESHOLD_MULTIPLIER в .env (например, с 1.5 до 1.0 или 0.8)"
            )


if __name__ == "__main__":
    asyncio.run(main())
