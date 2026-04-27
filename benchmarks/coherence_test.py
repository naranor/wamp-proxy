import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"

# Длинный технический текст, разбитый на части
TECH_SPEC = [
    "The system architecture consists of a distributed microservices cluster running on Kubernetes.",
    "Data persistence is handled by a PostgreSQL primary instance with two read-replicas for load balancing.",
    "For caching, we use Redis in-memory store with a cluster-mode enabled for high availability.",
    "Authentication is performed via an OAuth2 provider, issuing JWT tokens with a 1-hour expiration.",
    "The API Gateway is built using Kong, providing rate limiting and request transformation features.",
    "Monitoring is implemented using Prometheus for metrics and Grafana for visualization.",
    "Logging is centralized via ELK stack (Elasticsearch, Logstash, Kibana) for all production environments.",
    "Deployment pipelines are managed by GitLab CI, ensuring zero-downtime rolling updates.",
]


def generate_coherence_messages():
    messages = [
        {
            "role": "system",
            "content": "You are a technical writer. Summarize the following system architecture details into a cohesive paragraph.",
        }
    ]

    for i, part in enumerate(TECH_SPEC):
        messages.append({"role": "user", "content": f"Part {part}"})
        messages.append(
            {"role": "assistant", "content": f"Added part {i + 1} to the specification."}
        )

    messages.append(
        {
            "role": "user",
            "content": "Now, provide a comprehensive summary of the architecture. Mention the database, cache, and monitoring tools explicitly.",
        }
    )
    return messages


async def run_test(name, url, messages, model="auto"):
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
            print(f"🤖 Резюме: {answer.strip()[:300]}...")
            return tokens
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


async def main():
    print("=== BENCHMARK: Coherence & Summarization ===")
    messages = generate_coherence_messages()

    # Auto-detect model
    model_name = "auto"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get("http://192.168.92.2:8383/api/v1/models")
            model_name = res.json()["data"][0]["id"]
    except Exception:
        pass

    print(f"Общее кол-во сообщений: {len(messages)}")

    t_direct = await run_test("БЕЗ ПРОКСИ (Full Specs)", DIRECT_URL, messages, model=model_name)
    t_proxy = await run_test(
        "ЧЕРЕЗ ПРОКСИ (Compressed Specs)", PROXY_URL, messages, model=model_name
    )

    if t_direct and t_proxy:
        print(f"\n🎉 ЭКОНОМИЯ: {(1 - t_proxy / t_direct) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
