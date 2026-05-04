import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
import logging

def get_msg_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {'input_ids': ids, 'attention_mask': mask})
    
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, len(encoding.ids)]])

def train_message_router():
    print("=== TRAINING MULTI-CLASS MESSAGE CLASSIFIER ===")
    pruner = WAMPruner(model_dir="./model")
    
    # Categories: 0: Chatter, 1: Technical Fact, 2: Code, 3: Instruction
    data = [
        # 0: Chatter
        ("Hello, how are you today?", 0),
        ("I hope you are having a productive day!", 0),
        ("I am happy to help you with your request.", 0),
        ("Please let me know if you need anything else.", 0),
        ("Thanks for the info.", 0),
        
        # 1: Technical Fact
        ("The production database port is 5432.", 1),
        ("Dr. Elena Rossi is the lead scientist of Alpha.", 1),
        ("The API key is hidden in the .env file.", 1),
        ("We are using DeBERTa-v3-small model here.", 1),
        ("The server IP is 192.168.1.100.", 1),
        
        # 2: Code
        ("def main():\n    print('Hello World')", 2),
        ("import numpy as np\nfrom sklearn import log_reg", 2),
        ("{\"status\": \"ok\", \"port\": 5432}", 2),
        ("pipeline:\n  steps:\n    - build", 2),
        ("SELECT * FROM users WHERE id=1;", 2),
        
        # 3: Instruction
        ("Please analyze the differences between these two logs.", 3),
        ("Compare the budgets of project Alpha and Beta.", 3),
        ("Summarize the following technical documentation.", 3),
        ("Identify the root cause of the system failure.", 3),
        ("Rewrite the function to be more efficient.", 3)
    ] * 10 # Augment
    
    X, y = [], []
    print("Extracting message features...")
    for text, label in data:
        X.append(get_msg_features(pruner, text))
        y.append(label)
        
    # Multi-class Logistic Regression
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf.fit(X, y)
    
    print(f"✅ Training Accuracy: {clf.score(X, y)*100:.1f}%")
    
    # Save weights
    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "labels": ["Chatter", "Fact", "Code", "Instruction"]
    }
    
    with open("src/wamp/core/message_classifier_weights.json", "w") as f:
        json.dump(weights, f)
    
    print("🚀 Message classifier weights saved.")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_message_router()
