import os
from huggingface_hub import HfApi, upload_folder

token = os.getenv("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    exit(1)

api = HfApi()

print("Creating Space...")
api.create_repo(
    repo_id="Thanneermalai/india-itr-openenv",
    repo_type="space",
    space_sdk="gradio",
    token=token,
    exist_ok=True,
    private=False,
)
print("Space created!")

print("Uploading files...")
upload_folder(
    folder_path=".",
    repo_id="Thanneermalai/india-itr-openenv",
    repo_type="space",
    token=token,
    ignore_patterns=["*.pyc", "__pycache__", ".git", "push_to_hf.py"],
)
print("DONE! Visit: https://huggingface.co/spaces/Thanneermalai/india-itr-openenv")
