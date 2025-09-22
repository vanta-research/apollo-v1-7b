# Merging LoRA Weights with Base Model

This repository contains LoRA (Low-Rank Adaptation) weights that need to be merged with the base Mistral-7B-Instruct-v0.2 model for use.

## Quick Start

### Option 1: Use with Ollama (Recommended)
```bash
# Download this repository
git clone https://huggingface.co/vanta-research/apollo-v1-7b
cd apollo-v1-7b

# Create Ollama model
echo 'FROM mistral:7b' > Modelfile
ollama create apollo-v1-7b -f Modelfile
ollama run apollo-v1-7b
```

### Option 2: Merge with Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./apollo-v1-7b")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./apollo-v1-7b-merged")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.save_pretrained("./apollo-v1-7b-merged")
```

### Option 3: Use with PEFT directly
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "./apollo-v1-7b")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Use for inference
inputs = tokenizer("Hello, how can I help you today?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Requirements

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Python packages: `transformers`, `peft`, `torch`
- CUDA-compatible GPU (recommended)

## Model Architecture

- **Base Model**: Mistral 7B Instruct v0.2
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 64
- **Alpha**: 16
- **Target Modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj