# pip install huggingface_hub
from huggingface_hub import snapshot_download

local_dir = "models/LLaDA-8B-Instruct"
snapshot_download(
    repo_id="GSAI-ML/LLaDA-8B-Instruct",
    local_dir=local_dir,
    local_dir_use_symlinks=False,   # get real files (not symlinks)
    allow_patterns=[
        "*.safetensors","*.bin","*.pt",
        "config.json","tokenizer.*","*.model","special_tokens_map.json",
        "generation_config.json","*.py","*.json","*.txt"
    ],
)
print("Downloaded to:", local_dir)
