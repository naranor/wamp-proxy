import requests
from benchmarks.multi_doc_qa_long import generate_multi_doc_long


def test_live_proxy():
    print("=== LIVE PROXY TEST: REASONING SCENARIO ===")
    url = "http://localhost:3000/chat/completions"

    # Generate the 114 msgs context
    messages = generate_multi_doc_long()
    task = "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding and where is each project based?"

    # Append the final task
    messages.append({"role": "user", "content": task})

    payload = {"model": "auto", "messages": messages, "temperature": 0.0}

    print(f"Sending request with {len(messages)} messages to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=60)

        if not response.ok:
            print(f"\nError: {response.status_code} - {response.text}")
            return

        res_data = response.json()

        print("\n--- LLM RESPONSE ---")
        print(res_data["choices"][0]["message"]["content"])
        print("\n--------------------")

        # Check if the critical info is present in the answer
        content = res_data["choices"][0]["message"]["content"].lower()
        success = (
            "alpha" in content
            and "5 million" in content
            and "beta" in content
            and "3 million" in content
        )

        print(f"\nVerification: {'✅ SUCCESS' if success else '❌ FAILED'}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_live_proxy()
