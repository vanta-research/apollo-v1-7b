# Apollo V1 7B Usage Guide

## Installation & Setup

### Requirements
```bash
pip install transformers>=4.44.0 peft>=0.12.0 torch>=2.0.0
```

### Basic Setup
```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Load model (adjust device_map based on your hardware)
model = AutoPeftModelForCausalLM.from_pretrained(
    "vanta-research/apollo-v1-7b",
    torch_dtype=torch.float16,
    device_map="auto"  # or "cpu" for CPU-only
)

tokenizer = AutoTokenizer.from_pretrained("vanta-research/apollo-v1-7b")
```

## Usage Patterns

### 1. Mathematical Problem Solving

```python
def solve_math_problem(problem):
    prompt = f"Solve this step by step: {problem}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=400,
        temperature=0.1,  # Low temperature for accuracy
        do_sample=True,
        top_p=0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Examples
problems = [
    "What is 15% of 240?",
    "If x + 5 = 12, what is x?", 
    "A rectangle has length 8 and width 5. What is its area?"
]

for problem in problems:
    solution = solve_math_problem(problem)
    print(f"Problem: {problem}")
    print(f"Solution: {solution}")
    print("-" * 50)
```

### 2. Legal Reasoning

```python
def analyze_legal_scenario(scenario):
    prompt = f"Analyze this legal scenario: {scenario}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=600,
        temperature=0.2,  # Slightly higher for nuanced analysis
        repetition_penalty=1.1
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example legal scenarios
scenarios = [
    "A contract requires payment within 30 days, but the buyer received defective goods.",
    "Police conducted a search without a warrant, claiming exigent circumstances.",
    "An employee was fired for social media posts made outside work hours."
]

for scenario in scenarios:
    analysis = analyze_legal_scenario(scenario)
    print(f"Scenario: {scenario}")
    print(f"Analysis: {analysis}")
    print("-" * 50)
```

### 3. Logical Reasoning

```python
def solve_logic_puzzle(puzzle):
    prompt = f"Solve this logic puzzle step by step: {puzzle}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=500,
        temperature=0.1,
        top_k=50
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example logic puzzles
puzzles = [
    "If all A are B, and all B are C, what can we conclude about A and C?",
    "All cats are animals. Some animals are pets. Can we conclude all cats are pets?",
    "If it rains, the ground gets wet. The ground is wet. Did it rain?"
]

for puzzle in puzzles:
    solution = solve_logic_puzzle(puzzle)
    print(f"Puzzle: {puzzle}")
    print(f"Solution: {solution}")
    print("-" * 50)
```

## Advanced Usage

### Batch Processing
```python
def batch_process_questions(questions, batch_size=4):
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        # Process batch
        batch_results = []
        for question in batch:
            inputs = tokenizer(question, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=300)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            batch_results.append(response)
        
        results.extend(batch_results)
    
    return results
```

### Memory Optimization
```python
# For limited GPU memory
import torch

def memory_efficient_generation(prompt, max_length=400):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.1,
            use_cache=True,  # Enable KV caching
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Clear cache after generation
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Custom Prompting
```python
def create_apollo_prompt(question, context="", task_type="general"):
    """Create optimized prompts for different task types."""
    
    task_prompts = {
        "math": "Solve this mathematical problem step by step:",
        "legal": "Analyze this legal scenario considering relevant laws and precedents:",
        "logic": "Solve this logical reasoning problem step by step:",
        "general": "Please provide a clear and detailed response to:"
    }
    
    task_prompt = task_prompts.get(task_type, task_prompts["general"])
    
    if context:
        full_prompt = f"Context: {context}

{task_prompt} {question}"
    else:
        full_prompt = f"{task_prompt} {question}"
    
    return full_prompt

# Usage
question = "What is 25% of 160?"
prompt = create_apollo_prompt(question, task_type="math")
```

## Performance Optimization

### GPU Settings
```python
# For RTX 3060 (12GB) or similar
model = AutoPeftModelForCausalLM.from_pretrained(
    "vanta-research/apollo-v1-7b",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "10GB"}  # Reserve some GPU memory
)
```

### CPU Inference
```python
# For CPU-only inference
model = AutoPeftModelForCausalLM.from_pretrained(
    "vanta-research/apollo-v1-7b",
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu"
)
```

### Quantization (Coming Soon)
```python
# 8-bit quantization for reduced memory usage
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "vanta-research/apollo-v1-7b",
    quantization_config=quantization_config
)
```

## Integration Examples

### FastAPI Server
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    task_type: str = "general"
    max_length: int = 400

@app.post("/ask")
async def ask_apollo(request: QuestionRequest):
    prompt = create_apollo_prompt(request.question, task_type=request.task_type)
    response = memory_efficient_generation(prompt, request.max_length)
    
    return {
        "question": request.question,
        "response": response,
        "task_type": request.task_type
    }

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Gradio Interface
```python
import gradio as gr

def apollo_interface(message, task_type):
    prompt = create_apollo_prompt(message, task_type=task_type)
    return memory_efficient_generation(prompt)

interface = gr.Interface(
    fn=apollo_interface,
    inputs=[
        gr.Textbox(label="Your Question"),
        gr.Dropdown(["general", "math", "legal", "logic"], label="Task Type")
    ],
    outputs=gr.Textbox(label="Apollo's Response"),
    title="Apollo V1 7B Chat",
    description="Chat with Apollo V1 7B - Advanced Reasoning AI"
)

interface.launch(share=True)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, use CPU inference, or enable memory optimization
2. **Slow Generation**: Check device placement, enable caching, optimize prompt length
3. **Poor Quality**: Adjust temperature (lower for factual, higher for creative)

### Performance Tips

- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable gradient checkpointing for memory efficiency
- Use appropriate data types (float16 for GPU, float32 for CPU)
- Optimize prompt length and structure
- Consider quantization for resource-constrained environments

## Best Practices

1. **Prompt Engineering**: Be specific and clear in your questions
2. **Temperature Settings**: Use 0.1-0.2 for factual/mathematical tasks, 0.3-0.7 for creative tasks
3. **Context Management**: Provide relevant context for complex scenarios
4. **Verification**: Always verify critical information, especially for legal/financial advice
5. **Ethical Usage**: Use responsibly and within intended capabilities

For more examples and advanced usage patterns, see the GitHub repository and documentation.
