import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis

def audit_full_composite_router():
    print("=== FINAL LARGE-SCALE AUDIT: COMPOSITE ADAPTIVE ROUTER ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    test_suite = {
        "Needle": [
            "What is the port?", "Show me the secret key.", "What is the backup secret?",
            "API password?", "Find server IP.", "DB connection string.", "JWT expiry?",
            "What's the admin pass?", "Find lead scientist name.", "Budget of project Beta?"
        ],
        "Reasoning": [
            "Compare project Alpha and Beta in terms of budget.",
            "Analyze the differences between the current and future pipelines.",
            "Explain the link between Kong and OAuth2.",
            "Synthesize the team roles from fragments.", "Why did the database fail?",
            "Cross-reference logs with system events.", "Critique the security design.",
            "Which scientist has more experience?", "Evaluate the monitoring stack.",
            "Contrast monolith and microservices approach."
        ],
        "Summary": [
            "Give me a brief summary of our architecture discussion.",
            "Provide a comprehensive overview of all fragments.",
            "TL;DR of the whole chat history.", "Wrap up the spec.",
            "Technical synopsis of the project.", "Recap the infrastructure setup.",
            "Condensed version of our talk.", "Main takeaways from the chat?",
            "One paragraph summary please.", "Briefly outline the plan."
        ]
    }

    print(f"{'Category':<15} | {'Predicted':<12} | {'Status'} | {'Query'}")
    print("-" * 90)

    stats = {"Needle": {"c": 0, "t": 0}, "Reasoning": {"c": 0, "t": 0}, "Summary": {"c": 0, "t": 0}}
    
    for category, queries in test_suite.items():
        for query in queries:
            predicted = pruner.classify_task(query)
            
            is_correct = (category == predicted)
            if is_correct: stats[category]["c"] += 1
            stats[category]["t"] += 1
            
            print(f"{category:<15} | {predicted:<12} | {'✅' if is_correct else '❌'} | {query[:40]}")

    print("\n--- DETAILED ACCURACY ---")
    total_c, total_t = 0, 0
    for cat, s in stats.items():
        acc = (s["c"] / s["t"]) * 100
        print(f"{cat:<15}: {acc:>5.1f}% ({s['c']}/{s['t']})")
        total_c += s["c"]
        total_t += s["t"]
    
    print(f"\nOverall Router Accuracy: {total_c/total_t*100:.1f}%")

if __name__ == "__main__":
    import logging
    logging.getLogger("wamp").setLevel(logging.WARNING)
    audit_full_composite_router()
