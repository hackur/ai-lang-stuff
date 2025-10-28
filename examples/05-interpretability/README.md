# Mechanistic Interpretability Examples

This directory contains comprehensive examples for understanding transformer models using **TransformerLens**, a library for mechanistic interpretability developed by Neel Nanda and others.

## What is Mechanistic Interpretability?

Mechanistic interpretability aims to reverse-engineer neural networks to understand **how** they work, not just **what** they do. Instead of treating models as black boxes, we open them up to examine their internal computations.

### Key Concepts

**Attention Patterns**: How models decide which tokens to focus on
- Visualize which parts of the input each attention head looks at
- Identify specialized heads (induction, previous token, syntax)
- Understand information routing through the model

**Activation Patching**: Causal intervention to identify important components
- Replace activations from one run into another
- Measure how much behavior is restored
- Find which components are **causally** responsible for behaviors

**Computational Circuits**: Subgraphs that implement specific algorithms
- Chains of components that work together
- Example: induction circuit for in-context learning
- Necessary and sufficient for specific behaviors

## Prerequisites

Install the required packages:

```bash
# Core dependencies
uv add transformer-lens torch numpy

# Visualization
uv add plotly matplotlib circuitsvis

# Graph visualization (for circuits)
uv add networkx

# For Jupyter notebooks
uv add jupyter ipykernel
```

### System Requirements

- **GPU**: Recommended but not required (CPU/MPS works for small models)
- **RAM**: 8GB+ (16GB+ recommended for larger models)
- **Disk**: 2GB+ for model downloads
- **Python**: 3.9+

## Examples Overview

### 1. Attention Visualization (`attention_visualization.ipynb`)

**Jupyter notebook** for interactive exploration of attention patterns.

**What you'll learn:**
- Load models with TransformerLens
- Extract and visualize attention patterns
- Identify induction heads
- Analyze attention statistics (entropy, range, focus)
- Use CircuitsVis for interactive visualization

**Key features:**
- Interactive heatmaps with Plotly
- Grid view of all heads in a layer
- Induction head detection
- Attention pattern statistics
- Educational comments throughout

**How to run:**
```bash
# Start Jupyter
jupyter notebook attention_visualization.ipynb

# Or use Jupyter Lab
jupyter lab attention_visualization.ipynb
```

**Expected output:**
- Interactive attention heatmaps
- Induction head rankings
- Statistical analysis of attention patterns
- CircuitsVis visualizations

### 2. Activation Patching (`activation_patching.py`)

**Python script** demonstrating causal intervention analysis.

**What you'll learn:**
- Perform activation patching experiments
- Measure component importance causally
- Identify which heads/layers matter for specific tasks
- Distinguish correlation from causation

**Key features:**
- Indirect Object Identification (IOI) task
- Factual recall experiments
- Component importance heatmaps
- Top component identification

**How to run:**
```bash
python activation_patching.py
```

**Expected output:**
```
ACTIVATION PATCHING EXPERIMENT
======================================================================

Clean prompt: When Mary and John went to the store, Mary gave a drink to
Corrupted prompt: When John and Mary went to the store, John gave a drink to
Correct answer: ' John' (token 1757)
Incorrect answer: ' Mary' (token 5335)

1. Running baseline computations...
   Clean logit diff: 3.452 (prefers correct answer)
   Corrupted logit diff: -2.134

2. Patching attention heads...
   Layer 11/11 ✓

3. Patching MLP layers...
   Layer 11/11 ✓

Top 5 Attention Heads:
1. Layer  9, Head  9: 0.867 recovery
2. Layer  8, Head  6: 0.743 recovery
...
```

**Generated files:**
- `ioi_patching_results.png` - Heatmap of component importance
- `factual_patching_results.png` - Factual recall results

### 3. Circuit Discovery (`circuit_discovery.py`)

**Python script** for discovering computational circuits.

**What you'll learn:**
- Find computational circuits in models
- Trace information flow between components
- Analyze component functions
- Visualize circuits as graphs

**Key features:**
- Direct effect computation
- Connection strength analysis
- Circuit visualization with NetworkX
- Component function analysis
- Induction circuit demonstration
- Factual recall circuit

**How to run:**
```bash
python circuit_discovery.py
```

**Expected output:**
```
DISCOVERING INDUCTION CIRCUIT
======================================================================

Task: Complete repeated sequence [A][B]...[A] → [B]
This is the basic mechanism for in-context learning!

Prompt: The cat sat on the mat. The dog sat on the rug. The cat sat on the
Expected completion: 'mat' (repeating the pattern)

1. Identifying important components...
  Layer 11/11 ✓

Found 15 important components:
  L5H1: 2.134
  L6H9: 1.876
  L4H11: 1.654
...

2. Visualizing circuit...
Circuit visualization saved to: induction_circuit.png

3. Analyzing component functions...
L5H1 (importance: 2.134)
Most promoted tokens:
  'mat': +3.452
  'rug': +2.134
  ...
```

**Generated files:**
- `induction_circuit.png` - Graph visualization of induction circuit
- `factual_circuit.png` - Graph visualization of factual recall circuit

## Understanding the Results

### Attention Patterns

**What to look for:**

1. **Previous Token Heads**: Attend to the immediately preceding token
   - Show strong diagonal pattern
   - Common in early layers
   - Help with n-gram patterns

2. **Induction Heads**: Complete repeated patterns [A][B]...[A] → [B]
   - Typically in middle layers (5-9 in GPT-2)
   - Show diagonal stripe pattern on repeated sequences
   - Core mechanism for in-context learning

3. **Positional Heads**: Attend based on absolute/relative position
   - Attend to beginning of sequence (BOS token)
   - Show vertical or horizontal stripe patterns

4. **Syntax Heads**: Attend to syntactically related tokens
   - More complex, context-dependent patterns
   - Subject-verb agreement, noun-modifier relations

### Activation Patching Results

**Interpreting recovery scores:**

- **0.0**: Component has no effect on behavior
- **0.5**: Component accounts for 50% of the behavior difference
- **1.0**: Component fully restores clean behavior
- **Negative**: Component works against the target behavior

**Important heads** (high recovery):
- These are **causally important** for the task
- Patching them fixes the model's behavior
- Prime targets for further investigation

### Circuit Visualizations

**Graph structure:**

- **Nodes**: Model components (attention heads, MLPs)
- **Node color**: Importance (darker = more important)
- **Edges**: Information flow (thicker = stronger connection)
- **Layout**: Organized by layer (left to right)

**Circuit types:**

1. **Sequential circuits**: Linear chains of computation
2. **Parallel circuits**: Multiple pathways to same output
3. **Hierarchical circuits**: Earlier circuits feed later ones

## Common Tasks & Circuits

### 1. Induction (In-Context Learning)

**Task**: [A][B]...[A] → predict [B]

**Circuit components:**
- **Previous token heads** (early layers): Attend to previous token
- **Induction heads** (middle layers): Attend to token after previous occurrence
- **Information flow**: Previous token head → Induction head → Output

**Why it matters**: Foundation for in-context learning

### 2. Indirect Object Identification (IOI)

**Task**: "Mary and John went... Mary gave to" → "John"

**Circuit components:**
- **Duplicate token heads**: Identify repeated names
- **S-inhibition heads**: Suppress the subject name
- **Name mover heads**: Copy the correct name to output

**Why it matters**: Tests ability to track entities and relationships

### 3. Factual Recall

**Task**: "The Eiffel Tower is in" → "Paris"

**Circuit components:**
- **Entity recognition** (early layers): Identify the subject
- **Fact retrieval** (middle MLPs): Retrieve associated information
- **Output formatting** (late layers): Format the answer

**Why it matters**: Understanding how models store and retrieve knowledge

## Troubleshooting

### Model Download Fails

**Problem**: Model fails to download or times out

**Solution**:
```python
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/large/disk'

# Or download manually first
from transformers import GPT2LMHeadModel
GPT2LMHeadModel.from_pretrained("gpt2")

# Then use TransformerLens
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2-small")
```

### Out of Memory

**Problem**: CUDA/MPS out of memory errors

**Solutions**:
```python
# 1. Use smaller sequences
tokens = tokens[:, :30]  # Truncate to 30 tokens

# 2. Use CPU
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

# 3. Clear cache between runs
import torch
torch.cuda.empty_cache()  # For CUDA
# or
torch.mps.empty_cache()  # For MPS

# 4. Use smaller model
model = HookedTransformer.from_pretrained("gpt2-small")  # 124M params
# instead of gpt2-medium (355M) or gpt2-large (774M)
```

### Visualizations Not Showing

**Problem**: Plots don't appear in Jupyter or script

**Solutions**:

For Jupyter notebooks:
```python
# Add at top of notebook
%matplotlib inline

# For Plotly
import plotly.io as pio
pio.renderers.default = "notebook"
```

For scripts:
```python
# Explicitly show plots
import matplotlib.pyplot as plt
plt.show()

# Or save to file
plt.savefig("output.png")

# For Plotly
fig.show(renderer="browser")  # Opens in browser
fig.write_html("output.html")  # Save to file
```

### Slow Performance

**Problem**: Examples run very slowly

**Solutions**:
```python
# 1. Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Reduce number of heads analyzed
important = discovery.find_important_components(tokens, top_k=10)  # Instead of 20

# 3. Use shorter sequences
tokens = tokens[:, :20]

# 4. Cache results
_, cache = model.run_with_cache(tokens)
# Reuse cache instead of re-running model

# 5. Use smaller model
model = HookedTransformer.from_pretrained("gpt2-small")
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'transformer_lens'`

**Solution**:
```bash
# Install with uv
uv add transformer-lens

# Or with pip
pip install transformer-lens

# Verify installation
python -c "import transformer_lens; print(transformer_lens.__version__)"
```

## Advanced Topics

### Custom Prompts

Modify the examples to analyze your own prompts:

```python
# In activation_patching.py
clean_prompt = "Your clean prompt here"
corrupted_prompt = "Your corrupted prompt here"
correct_answer = " expected"
incorrect_answer = " alternative"

results = run_patching_experiment(
    model, clean_prompt, corrupted_prompt,
    correct_answer, incorrect_answer
)
```

### Analyzing Local Models

Use TransformerLens with Ollama models (requires conversion):

```python
# Note: TransformerLens primarily supports HuggingFace models
# For Ollama models, you'd need to:
# 1. Export Ollama model to HuggingFace format
# 2. Load with TransformerLens

# This is an advanced topic - see TransformerLens docs for details
```

### Path Patching

For more precise circuit discovery:

```python
# Patch specific paths through the model
# This is more advanced than simple activation patching
# See TransformerLens advanced tutorials
```

### Attribution Patching

For gradient-based component importance:

```python
# Use gradients to measure component importance
# More efficient than exhaustive patching
# See "Attribution Patching" paper by Anthropic
```

## Learning Path

**Recommended order:**

1. **Start**: `attention_visualization.ipynb`
   - Builds intuition for attention patterns
   - Interactive and visual
   - Safe to experiment with

2. **Next**: `activation_patching.py`
   - Learn causal intervention methods
   - Understand component importance
   - Quantitative analysis

3. **Advanced**: `circuit_discovery.py`
   - Discover full computational circuits
   - Trace information flow
   - Graph-based analysis

4. **Beyond**: Read research papers
   - "A Mathematical Framework for Transformer Circuits"
   - "In-context Learning and Induction Heads"
   - "Interpretability in the Wild"

## Key Concepts Reference

### Residual Stream

The main "highway" of information flow through the transformer:
- Each component reads from and writes to the residual stream
- Residual connections allow information to flow directly from input to output
- Circuit analysis traces how components modify the residual stream

### Attention Heads

Individual attention mechanisms within a layer:
- Each head can specialize in different patterns
- Heads work in parallel, outputs are combined
- Different heads implement different algorithms

### Direct vs Indirect Effects

- **Direct effect**: Component's contribution to final output
- **Indirect effect**: Component's contribution through other components
- Total effect = direct + indirect

### Logit Lens

Technique for understanding intermediate representations:
- Project intermediate activations to vocabulary
- See what the model "thinks" at each layer
- Used in component function analysis

## Further Resources

### Documentation
- [TransformerLens Docs](https://transformerlensorg.github.io/TransformerLens/)
- [TransformerLens GitHub](https://github.com/TransformerLensOrg/TransformerLens)

### Research Papers
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Interpretability in the Wild](https://arxiv.org/abs/2211.00593)

### Interactive Tools
- [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis)
- [TransformerLens Demos](https://transformerlensorg.github.io/TransformerLens/demos/Main_Demo.html)

### Community
- [Alignment Forum](https://www.alignmentforum.org/) - Discussion of interpretability research
- [LessWrong](https://www.lesswrong.com/) - AI alignment and interpretability
- [EleutherAI Discord](https://discord.gg/eleutherai) - Community discussion

## Contributing

Found a bug or have an improvement? Issues and PRs welcome!

**Common contributions:**
- New circuit discoveries
- Additional tasks/prompts
- Performance optimizations
- Better visualizations
- Documentation improvements

## License

These examples are part of the ai-lang-stuff project. See main repository for license details.

---

**Happy exploring!** Mechanistic interpretability is a rapidly evolving field. These examples provide a foundation, but there's much more to discover.
