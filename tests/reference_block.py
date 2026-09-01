import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "../models/SmolLM2-135M"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
)

model.eval()

input_ids = torch.tensor([[0, 1, 2]])

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        use_cache=False
    )

# logits последнего токена
logits = outputs.logits[0, -1]

print("logits:", logits.shape[0])

for i in range(10):
    print(logits[i].item())
