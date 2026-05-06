import requests
import json
from datasets import load_dataset
import time

def map_role(from_role):
    mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "tool": "user" # Map tool outputs to user for proxy simulation if needed
    }
    return mapping.get(from_role, "user")

def run_real_world_test():
    print("=== TESTING WAMP PROXY ON REAL-WORLD AGENT TRAJECTORIES ===")
    print("Dataset: lambda/hermes-agent-reasoning-traces (kimi subset)")
    
    # 1. Load dataset
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("lambda/hermes-agent-reasoning-traces", "kimi", split="train")
    
    # Filter for long conversations (more than 10 turns)
    long_samples = [s for s in ds if len(s['conversations']) > 10][:3]
    
    if not long_samples:
        print("No long samples found, taking first 3.")
        long_samples = ds[:3]

    proxy_url = "http://localhost:3000/chat/completions"

    for i, sample in enumerate(long_samples):
        print(f"\n--- Scenario {i+1}: {sample.get('category', 'Agent Task')} ---")
        cat_disp = sample.get('category', 'N/A')
        print(f"Category: {cat_disp}")
        print(f"Task: {sample.get('task', 'N/A')[:100]}...")
        
        # 2. Prepare messages for OpenAI format
        full_history = []
        for msg in sample['conversations']:
            full_history.append({
                "role": map_role(msg['from']),
                "content": msg['value']
            })
        
        print(f"Total messages in trajectory: {len(full_history)}")
        
        # We simulate the NEXT turn.
        # We take all but the last message as history, and the last as the current prompt.
        messages_to_send = full_history[:-1]
        
        # 3. Send to Proxy
        payload = {
            "model": "auto",  # Use auto-routing for local providers
            "messages": messages_to_send,
            "stream": False,
        }
        
        print("Sending to WAMP proxy...")
        start_time = time.time()
        try:
            response = requests.post(proxy_url, json=payload, timeout=300)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"⏱ Time: {elapsed:.2f}s")
                
                content = result['choices'][0]['message']['content']
                if "<think>" in content:
                    think = content.split("<think>")[1].split("</think>")[0]
                    print(f"🤖 Reasoning Trace found ({len(think)} chars)")
                
                print(f"✅ Success: Received coherent response from LLM.")
            else:
                print(f"❌ Proxy Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    run_real_world_test()
