import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_model(local_path, repo_id):
    api = HfApi()
    
    print(f"--- Uploading {local_path} to {repo_id} ---")
    
    # 1. Create repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} is ready.")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    # 2. Upload the entire folder
    try:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial export with WAMP attention weights (INT8 ONNX)"
        )
        print(f"🚀 Successfully uploaded to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload ONNX models to HuggingFace")
    parser.add_argument("--path", type=str, required=True, help="Local directory path")
    parser.add_argument("--repo", type=str, required=True, help="Target HuggingFace repo ID (e.g., naranor/model-name)")
    
    args = parser.parse_args()
    upload_model(args.path, args.repo)
