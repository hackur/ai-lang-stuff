# Skill: Model Comparison

## Purpose
Compare performance of multiple local models on identical tasks to help users choose the best model for their use case.

## Triggers
- User asks "which model should I use"
- User requests benchmark or comparison
- User mentions multiple models and performance
- User asks about model differences

## Process

### 1. Identify Comparison Criteria
Ask user which aspects matter most:
- Speed (inference time)
- Quality (accuracy, coherence)
- Resource usage (memory, CPU)
- Context window size
- Multimodal capabilities

### 2. Define Test Tasks
Create representative tasks:
- Code generation
- Explanation/reasoning
- Creative writing
- Factual Q&A
- Problem solving

### 3. Run Benchmark
```python
from examples.model_comparison import ModelBenchmark

benchmark = ModelBenchmark([
    "qwen3:8b",
    "qwen3:30b-a3b",
    "gemma3:4b",
    "gemma3:12b"
])

# Add relevant tasks
benchmark.run_task("Code Generation", prompt, expected_keywords)
benchmark.run_task("Reasoning", prompt, expected_keywords)

benchmark.save_results()
benchmark.print_summary()
```

### 4. Analyze Results
- Compare speed vs quality tradeoff
- Identify best model per task type
- Consider resource constraints

### 5. Make Recommendation
Provide clear recommendation with rationale:

**For general use**: qwen3:8b (balance of speed and quality)
**For complex tasks**: qwen3:30b-a3b (best quality, fast MoE)
**For resource-constrained**: gemma3:4b (minimal footprint)
**For multilingual**: gemma3:12b (140+ languages)

## Output
- Benchmark results table
- Visualization comparing models
- Clear recommendation based on use case
- Code to run chosen model

## Example Interaction

**User**: "Which model should I use for code generation?"

**Claude**:
"I'll compare models on code generation tasks. Let me run a quick benchmark..."

[Runs benchmark on code generation tasks]

"Results:
- qwen3:30b-a3b: 8.2s, Quality: 0.92 (Best quality, good speed)
- qwen3:8b: 3.5s, Quality: 0.87 (Great balance)
- gemma3:12b: 5.1s, Quality: 0.84
- gemma3:4b: 2.1s, Quality: 0.76 (Fastest, lower quality)

**Recommendation**: For code generation, I recommend **qwen3:30b-a3b**. While not the fastest, its MoE architecture provides excellent quality with reasonable speed. If speed is critical, qwen3:8b offers great results 2x faster."

[Provides code to use recommended model]
