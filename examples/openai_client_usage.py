from openai import OpenAI

# Initialize client pointing to WAMP-proxy
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="your-api-key-here",  # Proxy forwards this to upstream
)

# Example: Simple Chat Completion
# In a real scenario, the proxy will prune messages before forwarding to OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about robots."},
        {
            "role": "assistant",
            "content": "Why was the robot tired? Because it had a lot of 'data' processing!",
        },
        {"role": "user", "content": "Now tell me one about neural networks."},
    ],
    stream=False,
)

print("Response from proxy:")
print(response.choices[0].message.content)
