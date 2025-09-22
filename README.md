---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
tags:
- reasoning
- legal-analysis
- mathematical-reasoning
- logical-reasoning
- mistral
- lora
- vanta-research
- apollo
language:
- en
pipeline_tag: text-generation
---

# Apollo V1 7B

**Advanced Reasoning Language Model**

Apollo V1 7B is a specialized language model designed for advanced reasoning tasks, including logical reasoning, mathematical problem-solving, and legal analysis. Built on Mistral 7B-Instruct-v0.2 using LoRA fine-tuning, this model represents the first public release in the Apollo model series from VANTA Research.

## **Get the Complete Model**

**Download from HuggingFace**: The complete model with weights is available at:
**[https://huggingface.co/vanta-research/apollo-v1-7b](https://huggingface.co/vanta-research/apollo-v1-7b)**

*Note: Due to GitHub's file size limits, the large model files (adapter_model.safetensors, tokenizer files) are hosted on HuggingFace Hub.*

## Model Overview

Apollo V1 7B is a specialized language model optimized for reasoning-intensive tasks. The model demonstrates exceptional performance in logical reasoning, mathematical problem-solving, and legal analysis through targeted fine-tuning on curated reasoning datasets.

**Validated by VANTA Research Reasoning Evaluation (VRRE)**: Apollo V1 7B was comprehensively evaluated using our novel semantic framework that detects reasoning improvements invisible to standard benchmarks. VRRE revealed critical performance insights that traditional benchmarks missed entirely, establishing it as an essential tool for LLM reasoning assessment.

### Key Capabilities

- **Logical Reasoning**: Advanced syllogistic reasoning, conditional logic, and contradiction detection
- **Mathematical Problem Solving**: Step-by-step mathematical reasoning with high accuracy
- **Legal Analysis**: Educational legal reasoning and case analysis capabilities
- **High Performance**: Optimized for fast inference while maintaining quality
- **Consistent Identity**: Maintains clear model identity and capability awareness
- **VRRE Validated**: Proven performance through semantic reasoning evaluation

## Model Details

- **Base Model**: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Training Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **Parameters**: ~7.24B total parameters
- **LoRA Rank**: 64
- **Target Modules**: All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Training Precision**: 16-bit (bfloat16)
- **License**: Apache 2.0

## Quick Start

### Using the LoRA Adapter

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load and apply LoRA adapter from HuggingFace
model = PeftModel.from_pretrained(model, "vanta-research/apollo-v1-7b")

# Example usage
prompt = "Solve this logical reasoning problem: If all cats are mammals, and Fluffy is a cat, what can we conclude about Fluffy?"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Repository Contents

This GitHub repository contains:
- **Documentation**: Comprehensive model card, usage guides, and technical specifications
- **Configuration Files**: Model configuration and generation parameters
- **Merge Guide**: Instructions for creating merged model weights
- **License**: Apache 2.0 license for maximum adoption

**For the complete model weights**: Visit **[HuggingFace Hub](https://huggingface.co/vanta-research/apollo-v1-7b)**

## VRRE Framework Validation

Apollo V1 7B serves as a landmark validation of the VANTA Research Reasoning Evaluation (VRRE) framework:

- **Historic Discovery**: VRRE detected a 2.5x reasoning improvement between Apollo variants that standard benchmarks (BoolQ, PIQA, ARC) completely missed
- **Research Impact**: Demonstrates the critical need for semantic reasoning evaluation in AI development
- **Open Source**: Both Apollo V1 7B and VRRE are available for community use and validation

## License

This model is released under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.

## Links

- **HuggingFace Model**: [vanta-research/apollo-v1-7b](https://huggingface.co/vanta-research/apollo-v1-7b)
- **VRRE Framework**: [VANTA Research Reasoning Evaluation](https://github.com/vanta-research/reasoning-evaluation)
- **VANTA Research**: [alignmentstack.xyz](https://alignmentstack.xyz)

---

**Apollo V1 7B - Advancing the frontier of reasoning in language models**
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## License

This model is released under the Apache 2.0 License. See [LICENSE](./LICENSE) for details.

## Contact

For questions, issues, or collaboration opportunities, please visit the [model repository](https://huggingface.co/vanta-research/apollo-v1-7b).

---

**Apollo V1 7B - Advancing the frontier of reasoning in language models**
