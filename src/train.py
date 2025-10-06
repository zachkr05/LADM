import math, torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from pathlib import Path
from peft import LoraConfig, get_peft_model

def chat():
    local_repo = Path("~/LADM/src/models/LLaDA-8B-Instruct").expanduser().resolve()
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(
    root, trust_remote_code=True, local_files_only=True,
    torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)

    model.train()

    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decary=0.01)
    
    lora_cfg = LoraConfig(r=16, lora_alpha = 32, lora_dropout = 0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
     bias="none",
    task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.train()

if __name__ == "__main__":
    chat()

