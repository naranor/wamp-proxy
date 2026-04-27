import asyncio
import httpx
import time

PROXY_URL = "http://localhost:3000/chat/completions"
DIRECT_URL = "http://192.168.92.2:8383/api/v1/chat/completions"

TECH_SPEC = [
    "Architecture: distributed microservices cluster on Kubernetes.",
    "Database: PostgreSQL primary with two read-replicas.",
    "Cache: Redis in-memory store, cluster-mode enabled.",
    "Auth: OAuth2 provider issuing 1-hour JWT tokens.",
    "Gateway: Kong with rate limiting and transformations.",
    "Monitoring: Prometheus and Grafana.",
    "Logging: ELK stack (Elasticsearch, Logstash, Kibana).",
    "CI/CD: GitLab pipelines with zero-downtime rolling updates.",
]


def generate_coherence_long():
    messages = [
        {
            "role": "system",
            "content": "You are a technical writer. Summarize the following system details.",
        }
    ]
    for i, spec in enumerate(TECH_SPEC):
        messages.append({"role": "user", "content": f"Specification Part {i + 1}: {spec}"})
        messages.append({"role": "assistant", "content": "Confirmed."})
        # Heavy noise (6 pairs per part)
        for j in range(6):
            messages.append(
                {
                    "role": "user",
                    "content": f"Irrelevant discussion {i}-{j}: What about weather in city {i * j}?",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Weather in city {i * j} is not part of the spec, but let's focus on the technical details.",
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "Provide a comprehensive summary of the architecture. Mention the database, cache, and monitoring tools explicitly.",
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
            print(f"🤖 Summary: {answer.strip()[:200]}...")
            return tokens
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    print("=== BENCHMARK: Coherence & Summarization (Long: 100+ msgs) ===")
    messages = generate_coherence_long()
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
