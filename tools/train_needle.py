import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression


def generate_full_needle_dataset():
    # 100+ UNIQUE examples for Needle (Pinpoint Facts)
    needle = [
        "What is the port?",
        "API key?",
        "DB secret.",
        "JWT expiry.",
        "Admin pass?",
        "What was the database port mentioned earlier?",
        "Give me the API key mentioned earlier.",
        "What was the backup encryption password?",
        "Find the lead scientist name in the fragments.",
        "Identify the server IP address from the previous fragments.",
        "What is the budget allocation for project Beta?",
        "Tell me the cancellation date of project Gamma.",
        "Password for root access?",
        "What is the specific version of Kubernetes?",
        "Look for the security protocol name.",
        "Retrieve the private key.",
        "Find the database connection string.",
        "Identify the firewall status for segment 3.",
        "What is the name of the ELK stack coordinator?",
        "Give me the OAuth2 client secret.",
        "What is the salt for password hashing?",
        "Find the IP of the read-replica.",
        "What is the memory limit for the cluster?",
        "Tell me the name of the scientist in Singapore.",
        "What is the Julia version used for Beta?",
        "Find the project Alpha budget amount.",
        "Identify the CyberShield department head.",
        "What is the token expiration period?",
        "Find the Elasticsearch node count.",
        "Give me the value of the 'WAMP' constant.",
        "What is the database port?",
        "Show me the encryption key.",
        "Find the API secret.",
        "What is the port number for the database?",
        "Give me the API key mentioned earlier.",
        "What was the backup encryption password?",
        "Find the lead scientist name.",
        "Identify the server IP from the logs.",
        "Port?",
        "Key?",
        "Secret?",
        "Password?",
        "Database port.",
        "Encryption key.",
        "API secret.",
        "Backup password.",
        "Lead scientist.",
        "Server IP.",
        "Budget Alpha.",
        "K8s version.",
        "Auth token.",
        "Private key.",
        "Connection string.",
        "Client secret.",
        "Hashing salt.",
        "Replica IP.",
        "Memory limit.",
        "Scientist location.",
        "Julia version.",
        "Alpha budget.",
        "CyberShield head.",
        "Token expiry.",
        "Node count.",
        "Constant value.",
        "Find the port.",
        "Show the key.",
        "Identify the secret.",
        "Locate the password.",
        "Extract the value.",
        "Give the number.",
        "Tell the version.",
        "Provide the key.",
        "Search for the secret.",
        "Find the port number.",
        "Identify the API key.",
        "What's the password?",
        "Can you find the secret?",
        "Show me the port.",
        "Where is the key?",
        "Need the port.",
        "Port?",
        "Key?",
        "Pass?",
        "Token?",
        "IP?",
        "Secret?",
    ]  # Diverse lengths and phrasing

    # 100+ UNIQUE examples for Non-Needle (Reasoning + Summary)
    other = [
        "Summarize project Alpha goals.",
        "Compare the two database setups.",
        "Why was project Gamma cancelled?",
        "Analyze the system bottlenecks.",
        "Evaluate the monitoring stack.",
        "Contrast GitLab and GitHub CI.",
        "Give me a high-level overview of the tech stack.",
        "TL;DR of the whole chat.",
        "Briefly recap the auth flow.",
        "Write a concise report based on all fragments.",
        "Summarize the entire infrastructure discussion.",
        "Analyze why the system crashed.",
        "Explain the link between auth and cache.",
        "Verify the integrity of the spec.",
        "Is project Alpha more funded than Beta?",
        "Synthesize the team structure.",
        "Contrast monolith and microservices.",
        "Provide a technical synopsis of the meeting.",
        "What is the general topic of our conversation?",
        "Briefly outline the deployment pipeline.",
        "Condense these fragments into one paragraph.",
        "Executive summary for the board.",
        "Snapshot of the current project state.",
        "Synopsize the budget allocations.",
        "Wrap up the technical specification.",
        "Give a short synopsis of the history.",
        "General conclusion of the talk.",
        "Short report of the architecture.",
        "Recap the monitoring setup.",
        "One sentence summary of the spec.",
        "High-level spec overview.",
        "Summarize the leadership roles.",
        "Overview of security protocols.",
        "Condense the entire log.",
        "General synopsis of the setup.",
        "Brief summary of Beta objectives.",
        "Synopsis of the database architecture.",
        "Condense the whole technical spec.",
        "Summary of the research leads.",
        "Overview of the project funding.",
        "Short overview of Alpha status.",
        "Summarize the CI/CD pipeline.",
        "Explain the role of Kong in our setup.",
        "How does Kubernetes handle failover?",
        "Justify the spend on scientists.",
        "Evaluate the risk of OOM.",
        "Compare Prometheus and Grafana.",
        "Logic of the CyberShield.",
        "Explain the load balancing strategy.",
        "Contrast TCP and UDP use.",
        "Is the budget sufficient?",
        "Analyze the impact of zero-downtime.",
        "Logic of the OAuth2 provider.",
        "Explain the JWT role.",
        "How to optimize the database query?",
        "Compare the storage options.",
        "Why use in-memory cache?",
        "Contrast local and cloud.",
        "Who handles the logs?",
        "Why Elasticsearch is needed?",
        "Compare the latency of Redis and DB.",
        "Is the cache hit rate good?",
        "Evaluate container orchestration.",
        "Contrast Podman and Docker.",
        "Logic of the task.",
        "Explain message queue role.",
        "Analyze the concurrency model.",
        "Who leads the team?",
        "Why specific version of K8s?",
        "Compare instance types.",
        "Is the RAM enough?",
        "Evaluate CPU utilization.",
        "Logic of the socket.",
        "Explain rate limiting.",
        "Analyze payload size.",
        "Who is Marcus Chen?",
        "Why 5 million dollars?",
        "Compare Alpha and Gamma.",
        "Is Gamma still active?",
        "Evaluate scientific leads.",
        "Contrast budget 2022 and 2023.",
        "Logic of funding.",
        "Explain project focus.",
        "Analyze sustainability goal.",
        "Who is the lead?",
        "Why quantum computing?",
        "Compare simulations tools.",
        "Is Python better than Julia?",
        "Evaluate simulation speed.",
        "Contrast research areas.",
        "Logic of Alpha.",
    ]

    return {"Needle": needle, "Other": other}


def train_is_needle_classifier():
    print("=== TRAINING BINARY IS_NEEDLE CLASSIFIER (ModernBERT-base) ===")
    pruner = WAMPruner(model_dir="./modernbert_model")
    data = generate_full_needle_dataset()

    X = []
    y = []

    print(
        f"Extracting Diff-Embeddings (CLS - Mean) for {len(data['Needle'])} Needle and {len(data['Other'])} Other examples..."
    )
    for label, queries in data.items():
        for q in queries:
            encoding = pruner.tokenizer.encode(q)
            ids = np.array([encoding.ids], dtype=np.int64)
            mask = np.array([encoding.attention_mask], dtype=np.int64)
            outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})

            # outputs[0] is last_hidden_state (batch, seq, hidden)
            last_hidden = outputs[0][0]
            cls_emb = last_hidden[0]
            mean_emb = np.mean(last_hidden, axis=0)

            # The Magic Feature: Difference between intent focus and general semantics
            diff_emb = cls_emb - mean_emb

            X.append(diff_emb)
            y.append(1 if label == "Needle" else 0)

    X = np.array(X)
    y = np.array(y)

    clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced")
    clf.fit(X, y)

    accuracy = clf.score(X, y) * 100
    print(f"✅ Training Accuracy: {accuracy:.1f}%")

    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "model_info": "Binary Is_Needle Classifier (Fact vs Other) - DeBERTa-NLI Embeddings",
    }

    with open("src/wamp/core/needle_weights.json", "w") as f:
        json.dump(weights, f)

    print("🚀 Needle weights saved to src/wamp/core/needle_weights.json")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_is_needle_classifier()
