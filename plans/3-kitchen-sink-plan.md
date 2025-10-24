# Kitchen Sink Plan: Common Use Cases & Practical Examples

## Overview
This document provides concrete, runnable examples for common AI development tasks using local models, LangChain, LangGraph, MCP servers, and mechanistic interpretability tools. Every example is designed to run entirely locally without cloud dependencies.

---

## Table of Contents
1. [Basic Local LLM Interaction](#1-basic-local-llm-interaction)
2. [Agent with Tool Calling via MCP](#2-agent-with-tool-calling-via-mcp)
3. [Multi-Agent Research Pipeline](#3-multi-agent-research-pipeline)
4. [RAG System for Document QA](#4-rag-system-for-document-qa)
5. [Vision Model for Document Analysis](#5-vision-model-for-document-analysis)
6. [Mechanistic Interpretability Analysis](#6-mechanistic-interpretability-analysis)
7. [LangGraph State Machine with Persistence](#7-langgraph-state-machine-with-persistence)
8. [Model Comparison Framework](#8-model-comparison-framework)
9. [Fine-Tuning Local Models](#9-fine-tuning-local-models)
10. [Production Deployment Pattern](#10-production-deployment-pattern)

---

## 1. Basic Local LLM Interaction

### Purpose
Get started with local models using Ollama and LangChain. This is the "Hello World" of local AI development.

### Prerequisites
```bash
# Install Ollama
brew install ollama

# Pull models
ollama pull qwen3:8b
ollama pull gemma3:4b

# Start Ollama server
ollama serve
```

### Python Example: Simple Chat
```python
# examples/01-foundation/simple_chat.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize local model
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Explain list comprehensions in Python with an example.")
]

# Get response
response = llm.invoke(messages)
print(response.content)
```

### Node.js Example (Alternative)
```javascript
// examples/01-foundation/simple_chat.js
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  modelName: "qwen3:8b",
  temperature: 0.7,
  configuration: {
    baseURL: "http://localhost:11434/v1",
  },
});

const response = await llm.invoke("Explain async/await in JavaScript");
console.log(response.content);
```

### Expected Output
Clear explanation of the requested topic in 3-5 paragraphs with code examples.

### Common Issues
- **Ollama not running**: Run `ollama serve` in separate terminal
- **Model not found**: Run `ollama pull <model-name>` first
- **Port conflict**: Check `lsof -i :11434` and kill conflicting process

---

## 2. Agent with Tool Calling via MCP

### Purpose
Create an agent that can use external tools (file system, GitHub, web search) through Model Context Protocol servers.

### Setup MCP Servers
```bash
# Install official MCP servers
npx @modelcontextprotocol/create-server filesystem --path ./
npx @modelcontextprotocol/create-server github
```

### Python Example: File System Agent
```python
# examples/02-mcp/filesystem_agent.py
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
import subprocess
import json

# Define MCP tool wrapper
def call_mcp_tool(tool_name: str, params: dict) -> str:
    """Call an MCP server tool"""
    result = subprocess.run(
        ["npx", "@modelcontextprotocol/client", tool_name, json.dumps(params)],
        capture_output=True,
        text=True
    )
    return result.stdout

# Create tools
filesystem_tool = Tool(
    name="read_file",
    description="Read contents of a file. Input should be file path.",
    func=lambda path: call_mcp_tool("filesystem:read", {"path": path})
)

list_dir_tool = Tool(
    name="list_directory",
    description="List files in a directory. Input should be directory path.",
    func=lambda path: call_mcp_tool("filesystem:list", {"path": path})
)

# Create agent
llm = ChatOllama(model="qwen3:30b-a3b", base_url="http://localhost:11434")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful file system assistant. Use tools to access files."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [filesystem_tool, list_dir_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[filesystem_tool, list_dir_tool])

# Execute
result = agent_executor.invoke({
    "input": "What Python files exist in the current directory and what's in main.py?"
})
print(result["output"])
```

### Expected Behavior
Agent will:
1. Use list_directory tool to find Python files
2. Use read_file tool to read main.py
3. Summarize findings in natural language

### Checklist
- [ ] MCP servers installed and accessible
- [ ] Agent can successfully call tools
- [ ] Agent chains multiple tool calls correctly
- [ ] Error handling for missing files works
- [ ] Response is coherent and accurate

---

## 3. Multi-Agent Research Pipeline

### Purpose
Demonstrate LangGraph's orchestration capabilities with multiple specialized agents working together.

### Architecture
```
[Researcher] -> [Analyzer] -> [Writer] -> [Reviewer]
     |              |            |           |
   Search        Synthesize   Generate    Quality
   Sources        Data        Report      Check
```

### Python Example: Research Pipeline
```python
# examples/03-multi-agent/research_pipeline.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import operator

# Define state
class ResearchState(TypedDict):
    topic: str
    sources: Annotated[List[str], operator.add]
    analysis: str
    report: str
    approved: bool

# Initialize models
researcher_llm = ChatOllama(model="qwen3:8b")
analyst_llm = ChatOllama(model="qwen3:30b-a3b")
writer_llm = ChatOllama(model="gemma3:12b")
reviewer_llm = ChatOllama(model="qwen3:8b")

# Define agents
def researcher(state: ResearchState) -> ResearchState:
    """Find relevant sources"""
    prompt = f"List 5 key topics to research about: {state['topic']}"
    response = researcher_llm.invoke([HumanMessage(content=prompt)])
    sources = response.content.split("\n")
    return {"sources": sources}

def analyst(state: ResearchState) -> ResearchState:
    """Analyze gathered information"""
    prompt = f"Analyze these research topics:\n{chr(10).join(state['sources'])}\nProvide key insights."
    response = analyst_llm.invoke([HumanMessage(content=prompt)])
    return {"analysis": response.content}

def writer(state: ResearchState) -> ResearchState:
    """Generate report"""
    prompt = f"Write a comprehensive report on {state['topic']} based on:\n{state['analysis']}"
    response = writer_llm.invoke([HumanMessage(content=prompt)])
    return {"report": response.content}

def reviewer(state: ResearchState) -> ResearchState:
    """Review quality"""
    prompt = f"Review this report for quality:\n{state['report']}\nIs it comprehensive? (yes/no)"
    response = reviewer_llm.invoke([HumanMessage(content=prompt)])
    approved = "yes" in response.content.lower()
    return {"approved": approved}

# Build graph
workflow = StateGraph(ResearchState)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", "reviewer")

# Conditional routing based on review
def check_approval(state: ResearchState) -> str:
    return "end" if state["approved"] else "writer"

workflow.add_conditional_edges("reviewer", check_approval, {"writer": "writer", "end": END})

# Compile and run
app = workflow.compile()

# Execute
initial_state = {"topic": "mechanistic interpretability in transformers"}
result = app.invoke(initial_state)
print("Final Report:", result["report"])
```

### Visual Workflow (LangGraph Studio)
```bash
# Launch LangGraph Studio to visualize
npx langgraph@latest dev
# Open http://localhost:3000
# Load examples/03-multi-agent/research_pipeline.py
```

### Checklist
- [ ] All four agents execute in sequence
- [ ] State persists across agent transitions
- [ ] Conditional routing based on review works
- [ ] Failed reviews trigger re-writing
- [ ] Final report is comprehensive and well-structured

---

## 4. RAG System for Document QA

### Purpose
Build a retrieval-augmented generation system for answering questions from large document collections.

### Setup Vector Store
```bash
# Install dependencies in pyproject.toml
uv add chromadb langchain-chroma sentence-transformers
```

### Python Example: Document QA System
```python
# examples/04-rag/document_qa.py
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA

# Load documents
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_documents(documents)

# Create embeddings using local model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding",
    base_url="http://localhost:11434"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./data/chroma_db"
)

# Create QA chain
llm = ChatOllama(model="qwen3:30b-a3b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

# Query
query = "What are the key features of LangGraph?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}")
```

### Advanced: Re-ranking
```python
# Add re-ranking for better results
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Use compressed retriever in QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

### Checklist
- [ ] Documents successfully loaded and chunked
- [ ] Embeddings generated and stored in Chroma
- [ ] Retrieval returns relevant chunks
- [ ] LLM synthesizes accurate answers from chunks
- [ ] Source attribution is correct
- [ ] Re-ranking improves relevance (optional)

---

## 5. Vision Model for Document Analysis

### Purpose
Use Qwen3-VL to analyze images, diagrams, and scanned documents.

### Setup
```bash
# Pull vision model
ollama pull qwen3-vl:8b
```

### Python Example: Image Understanding
```python
# examples/04-rag/vision_analysis.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64

def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Initialize vision model
vision_llm = ChatOllama(model="qwen3-vl:8b")

# Load and encode image
image_path = "./data/diagram.png"
image_data = encode_image(image_path)

# Create message with image
message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this diagram in detail. What are the key components?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        },
    ],
)

# Get response
response = vision_llm.invoke([message])
print(response.content)
```

### Multi-Page Document Processing
```python
# Process multi-page PDF with vision
from pdf2image import convert_from_path

def analyze_pdf_with_vision(pdf_path: str) -> List[str]:
    """Analyze each page of PDF with vision model"""
    pages = convert_from_path(pdf_path)
    analyses = []

    for i, page in enumerate(pages):
        # Save page as temp image
        page_path = f"/tmp/page_{i}.png"
        page.save(page_path, "PNG")

        # Analyze with vision model
        image_data = encode_image(page_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract all text and describe any diagrams:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        )
        response = vision_llm.invoke([message])
        analyses.append(response.content)

    return analyses

# Use
analyses = analyze_pdf_with_vision("./data/research_paper.pdf")
for i, analysis in enumerate(analyses):
    print(f"Page {i+1}:", analysis[:200], "...")
```

### Checklist
- [ ] Vision model correctly identifies objects in images
- [ ] Text extraction from images works accurately
- [ ] Diagram descriptions are comprehensive
- [ ] Multi-page document processing completes
- [ ] Combined vision + text RAG pipeline functions

---

## 6. Mechanistic Interpretability Analysis

### Purpose
Analyze model internals using TransformerLens to understand how models process information.

### Setup
```bash
uv add transformer-lens torch numpy matplotlib plotly
```

### Python Example: Attention Pattern Analysis
```python
# examples/05-interpretability/attention_analysis.py
import torch
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import numpy as np

# Load model for analysis (smaller local model)
model = HookedTransformer.from_pretrained("gpt2-small")

# Define prompt
prompt = "The cat sat on the"
tokens = model.to_tokens(prompt)

# Run with cache to capture activations
logits, cache = model.run_with_cache(tokens)

# Extract attention patterns from layer 6
attention_pattern = cache["pattern", 6]  # [batch, head, query_pos, key_pos]

# Visualize attention for head 0
head_0_attention = attention_pattern[0, 0].detach().cpu().numpy()

plt.figure(figsize=(10, 8))
plt.imshow(head_0_attention, cmap="Blues")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.title("Attention Pattern - Layer 6, Head 0")
plt.colorbar()
plt.savefig("attention_pattern.png")
print("Attention pattern saved to attention_pattern.png")

# Analyze which heads focus on recent tokens vs distant context
def analyze_attention_locality(cache, layer_idx: int):
    """Measure how local vs global each attention head is"""
    attention = cache["pattern", layer_idx][0]  # [head, query, key]
    n_heads = attention.shape[0]

    locality_scores = []
    for head in range(n_heads):
        head_attn = attention[head]
        # Calculate average distance of attended tokens
        positions = torch.arange(head_attn.shape[0])
        attended_distance = (head_attn * positions.unsqueeze(0)).sum(dim=1)
        avg_distance = attended_distance.mean().item()
        locality_scores.append(avg_distance)

    return locality_scores

locality = analyze_attention_locality(cache, 6)
print(f"Attention locality scores for layer 6: {locality}")
```

### Circuit Discovery
```python
# examples/05-interpretability/circuit_discovery.py
from transformer_lens import HookedTransformer, ActivationCache
import torch

model = HookedTransformer.from_pretrained("gpt2-small")

def ablate_component(cache: ActivationCache, component: str, position: int):
    """Zero out a specific component to test its importance"""
    def ablation_hook(activation, hook):
        activation[:, position, :] = 0
        return activation
    return ablation_hook

# Test prompt
prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

# Get baseline prediction
baseline_logits = model(tokens)
baseline_prob = torch.softmax(baseline_logits[0, -1], dim=0)
mary_token = model.to_single_token(" Mary")
john_token = model.to_single_token(" John")

print(f"Baseline - Mary: {baseline_prob[mary_token]:.3f}, John: {baseline_prob[john_token]:.3f}")

# Ablate each layer and measure effect
for layer in range(model.cfg.n_layers):
    model.reset_hooks()
    hook_name = f"blocks.{layer}.hook_resid_post"
    model.add_hook(hook_name, ablate_component(None, "residual", -1))

    ablated_logits = model(tokens)
    ablated_prob = torch.softmax(ablated_logits[0, -1], dim=0)

    print(f"Layer {layer} ablated - Mary: {ablated_prob[mary_token]:.3f}, John: {ablated_prob[john_token]:.3f}")
    model.reset_hooks()
```

### Checklist
- [ ] Model loads successfully with TransformerLens
- [ ] Attention patterns extracted and visualized
- [ ] Locality analysis identifies local vs global heads
- [ ] Ablation studies show component importance
- [ ] Circuit discovery identifies key components for specific behaviors

---

## 7. LangGraph State Machine with Persistence

### Purpose
Build a sophisticated multi-turn conversation agent with persistent memory across sessions.

### Python Example: Stateful Customer Service Agent
```python
# examples/03-multi-agent/stateful_agent.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
import sqlite3

# Define state
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_info: dict
    issue_resolved: bool
    escalate: bool

# Initialize
llm = ChatOllama(model="qwen3:30b-a3b")

# Create checkpoint saver
conn = sqlite3.connect("./data/checkpoints.db")
checkpointer = SqliteSaver(conn)

# Define nodes
def understand_query(state: ConversationState) -> ConversationState:
    """Understand user's issue"""
    last_message = state["messages"][-1]

    prompt = f"""Analyze this customer query: {last_message.content}
    Extract: issue type, urgency, user sentiment.
    Respond in JSON format."""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse user info (simplified)
    user_info = {"query_type": "general", "urgency": "normal"}

    return {"user_info": user_info}

def provide_solution(state: ConversationState) -> ConversationState:
    """Provide solution based on context"""
    conversation = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])

    prompt = f"""Based on this conversation:
    {conversation}

    User info: {state['user_info']}

    Provide a helpful solution."""

    response = llm.invoke([HumanMessage(content=prompt)])
    messages = [AIMessage(content=response.content)]

    return {"messages": messages}

def check_resolution(state: ConversationState) -> ConversationState:
    """Check if issue is resolved"""
    last_message = state["messages"][-1]

    prompt = f"Does this response solve the issue? (yes/no): {last_message.content}"
    response = llm.invoke([HumanMessage(content=prompt)])

    resolved = "yes" in response.content.lower()
    escalate = "escalate" in last_message.content.lower()

    return {"issue_resolved": resolved, "escalate": escalate}

# Build graph
workflow = StateGraph(ConversationState)
workflow.add_node("understand", understand_query)
workflow.add_node("solve", provide_solution)
workflow.add_node("check", check_resolution)

workflow.set_entry_point("understand")
workflow.add_edge("understand", "solve")
workflow.add_edge("solve", "check")

# Conditional routing
def route_after_check(state: ConversationState) -> str:
    if state.get("escalate"):
        return "escalate"
    elif state.get("issue_resolved"):
        return "end"
    else:
        return "understand"

workflow.add_conditional_edges(
    "check",
    route_after_check,
    {"understand": "understand", "escalate": END, "end": END}
)

# Compile with checkpointer
app = workflow.compile(checkpointer=checkpointer)

# Use with session persistence
config = {"configurable": {"thread_id": "user_123"}}

# First interaction
result = app.invoke({
    "messages": [HumanMessage(content="My order hasn't arrived yet.")],
    "user_info": {},
    "issue_resolved": False,
    "escalate": False
}, config)

print("Agent response:", result["messages"][-1].content)

# Later interaction (same thread)
result = app.invoke({
    "messages": [HumanMessage(content="It was supposed to arrive yesterday.")],
}, config)

print("Follow-up response:", result["messages"][-1].content)
```

### Checklist
- [ ] State persists across multiple invocations
- [ ] SQLite checkpoint system works correctly
- [ ] Thread IDs isolate different conversations
- [ ] Conditional routing based on state works
- [ ] Agent maintains context across turns

---

## 8. Model Comparison Framework

### Purpose
Systematically compare performance of different local models on identical tasks.

### Python Example: Benchmark Suite
```python
# examples/06-production/model_comparison.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import time
import json
from typing import List, Dict

class ModelBenchmark:
    def __init__(self, models: List[str]):
        self.models = {name: ChatOllama(model=name) for name in models}
        self.results = []

    def run_task(self, task_name: str, prompt: str, expected_keywords: List[str] = None):
        """Run a task on all models and collect results"""
        print(f"\nRunning task: {task_name}")
        task_results = {"task": task_name, "models": {}}

        for model_name, model in self.models.items():
            start_time = time.time()

            try:
                response = model.invoke([HumanMessage(content=prompt)])
                elapsed = time.time() - start_time

                # Calculate quality score
                quality_score = 0
                if expected_keywords:
                    quality_score = sum(
                        1 for kw in expected_keywords
                        if kw.lower() in response.content.lower()
                    ) / len(expected_keywords)

                task_results["models"][model_name] = {
                    "response": response.content[:200],  # Truncate for display
                    "time": elapsed,
                    "quality_score": quality_score,
                    "length": len(response.content),
                }

                print(f"{model_name}: {elapsed:.2f}s, quality: {quality_score:.2f}")

            except Exception as e:
                task_results["models"][model_name] = {"error": str(e)}
                print(f"{model_name}: ERROR - {e}")

        self.results.append(task_results)
        return task_results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to file"""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        for model_name in self.models.keys():
            avg_time = sum(
                r["models"][model_name].get("time", 0)
                for r in self.results
            ) / len(self.results)

            avg_quality = sum(
                r["models"][model_name].get("quality_score", 0)
                for r in self.results
            ) / len(self.results)

            print(f"\n{model_name}:")
            print(f"  Average Time: {avg_time:.2f}s")
            print(f"  Average Quality: {avg_quality:.2f}")

# Run benchmark
benchmark = ModelBenchmark([
    "qwen3:8b",
    "qwen3:30b-a3b",
    "gemma3:4b",
    "gemma3:12b",
])

# Task 1: Code generation
benchmark.run_task(
    "Code Generation",
    "Write a Python function to calculate fibonacci numbers with memoization.",
    expected_keywords=["def", "fibonacci", "memo", "cache", "return"]
)

# Task 2: Explanation
benchmark.run_task(
    "Concept Explanation",
    "Explain the concept of attention mechanisms in transformers.",
    expected_keywords=["attention", "query", "key", "value", "softmax"]
)

# Task 3: Reasoning
benchmark.run_task(
    "Logical Reasoning",
    "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
    expected_keywords=["cannot", "conclude", "logic", "some"]
)

# Task 4: Creative writing
benchmark.run_task(
    "Creative Writing",
    "Write a haiku about artificial intelligence.",
    expected_keywords=["haiku", "syllable", "5-7-5"]
)

benchmark.save_results()
benchmark.print_summary()
```

### Visualization Dashboard
```python
# examples/06-production/visualize_benchmarks.py
import json
import matplotlib.pyplot as plt
import numpy as np

with open("benchmark_results.json") as f:
    results = json.load(f)

models = list(results[0]["models"].keys())
tasks = [r["task"] for r in results]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Speed comparison
times = np.array([
    [r["models"][m].get("time", 0) for m in models]
    for r in results
])

x = np.arange(len(tasks))
width = 0.2
for i, model in enumerate(models):
    ax1.bar(x + i*width, times[:, i], width, label=model)

ax1.set_xlabel('Task')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Speed Comparison')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(tasks, rotation=45, ha="right")
ax1.legend()

# Plot 2: Quality comparison
quality = np.array([
    [r["models"][m].get("quality_score", 0) for m in models]
    for r in results
])

for i, model in enumerate(models):
    ax2.bar(x + i*width, quality[:, i], width, label=model)

ax2.set_xlabel('Task')
ax2.set_ylabel('Quality Score')
ax2.set_title('Quality Comparison')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(tasks, rotation=45, ha="right")
ax2.legend()

plt.tight_layout()
plt.savefig("benchmark_comparison.png")
print("Visualization saved to benchmark_comparison.png")
```

### Checklist
- [ ] All models complete all tasks
- [ ] Timing measurements are accurate
- [ ] Quality scoring reflects task requirements
- [ ] Results saved in structured format
- [ ] Visualization shows clear comparisons
- [ ] Summary statistics calculated correctly

---

## 9. Fine-Tuning Local Models

### Purpose
Fine-tune a small local model on custom data for specialized tasks.

### Setup
```bash
uv add transformers datasets peft accelerate bitsandbytes
```

### Python Example: LoRA Fine-Tuning
```python
# examples/06-production/finetune_model.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# Load base model (small local model)
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Prepare dataset
dataset = load_dataset("json", data_files="./data/training_data.jsonl")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/finetuned_qwen",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

# Save fine-tuned model
model.save_pretrained("./models/finetuned_qwen")
tokenizer.save_pretrained("./models/finetuned_qwen")

print("Fine-tuning complete!")
```

### Using Fine-Tuned Model with Ollama
```bash
# Create Modelfile
cat > Modelfile <<EOF
FROM ./models/finetuned_qwen
PARAMETER temperature 0.7
EOF

# Create Ollama model
ollama create my-finetuned-model -f Modelfile

# Use it
ollama run my-finetuned-model
```

### Checklist
- [ ] Base model loads successfully
- [ ] LoRA configuration applied
- [ ] Training dataset formatted correctly
- [ ] Training completes without errors
- [ ] Fine-tuned model shows improved performance on target task
- [ ] Model exported to Ollama format

---

## 10. Production Deployment Pattern

### Purpose
Transform experimental notebooks into production-ready applications with proper error handling, logging, and configuration.

### Production Structure
```
production_agent/
├── config/
│   ├── settings.py
│   └── models.yaml
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── specialized.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── mcp_tools.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── retry.py
│   └── main.py
├── tests/
│   ├── test_agents.py
│   └── test_tools.py
├── pyproject.toml
└── README.md
```

### Python Example: Production Agent
```python
# src/main.py
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
import yaml
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class AgentConfig(BaseModel):
    model_name: str = Field(..., description="Name of the Ollama model")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_iterations: int = Field(10, ge=1)
    base_url: str = "http://localhost:11434"

def load_config(config_path: str = "config/models.yaml") -> AgentConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return AgentConfig(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.info("Using default configuration")
        return AgentConfig(model_name="qwen3:8b")

class ProductionAgent:
    """Production-ready agent with error handling and logging"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = self._initialize_llm()
        logger.info(f"Agent initialized with model: {config.model_name}")

    def _initialize_llm(self) -> ChatOllama:
        """Initialize LLM with retry logic"""
        try:
            llm = ChatOllama(
                model=self.config.model_name,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
            )
            # Test connection
            llm.invoke("test")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process(self, input_text: str) -> Optional[str]:
        """Process input with retry logic"""
        try:
            logger.info(f"Processing input: {input_text[:100]}...")
            response = self.llm.invoke(input_text)
            logger.info("Processing successful")
            return response.content
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def batch_process(self, inputs: list[str]) -> list[str]:
        """Process multiple inputs with error handling"""
        results = []
        for i, input_text in enumerate(inputs):
            try:
                result = self.process(input_text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process input {i}: {e}")
                results.append(None)

        success_rate = sum(1 for r in results if r is not None) / len(results)
        logger.info(f"Batch processing complete. Success rate: {success_rate:.2%}")
        return results

# CLI Interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Production Agent")
    parser.add_argument("--config", default="config/models.yaml", help="Config file path")
    parser.add_argument("--input", required=True, help="Input text or file path")
    parser.add_argument("--output", help="Output file path (optional)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize agent
    agent = ProductionAgent(config)

    # Process input
    if Path(args.input).exists():
        with open(args.input) as f:
            inputs = [line.strip() for line in f if line.strip()]
        results = agent.batch_process(inputs)
    else:
        results = [agent.process(args.input)]

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            for result in results:
                f.write(f"{result}\n\n")
        logger.info(f"Results saved to {args.output}")
    else:
        for result in results:
            print(result)

if __name__ == "__main__":
    main()
```

### Configuration File
```yaml
# config/models.yaml
model_name: "qwen3:30b-a3b"
temperature: 0.7
max_iterations: 10
base_url: "http://localhost:11434"

# Monitoring
langsmith:
  enabled: true
  project: "production-agent"

# Logging
logging:
  level: "INFO"
  file: "agent.log"
```

### Testing
```python
# tests/test_agents.py
import pytest
from src.main import ProductionAgent, AgentConfig

@pytest.fixture
def agent():
    config = AgentConfig(model_name="qwen3:8b")
    return ProductionAgent(config)

def test_agent_initialization(agent):
    assert agent.llm is not None
    assert agent.config.model_name == "qwen3:8b"

def test_process(agent):
    result = agent.process("What is 2+2?")
    assert result is not None
    assert "4" in result.lower()

def test_batch_process(agent):
    inputs = ["What is 2+2?", "What is 3+3?"]
    results = agent.batch_process(inputs)
    assert len(results) == 2
    assert all(r is not None for r in results)
```

### Deployment Script
```bash
#!/bin/bash
# scripts/deploy.sh

set -e

echo "Deploying production agent..."

# 1. Check dependencies
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama not installed"
    exit 1
fi

# 2. Install Python dependencies
uv sync

# 3. Pull required models
ollama pull qwen3:30b-a3b

# 4. Run tests
uv run pytest tests/

# 5. Start Ollama server
ollama serve &

# 6. Run agent
uv run python src/main.py --config config/models.yaml --input "$1" --output "$2"

echo "Deployment complete!"
```

### Checklist
- [ ] Configuration management implemented
- [ ] Comprehensive error handling in place
- [ ] Logging to file and console
- [ ] Retry logic for transient failures
- [ ] Unit tests cover main functionality
- [ ] CLI interface for easy usage
- [ ] Deployment script automates setup
- [ ] Documentation includes usage examples
- [ ] Monitoring and observability configured

---

## Summary of Examples

| Example | Purpose | Complexity | Time to Implement |
|---------|---------|------------|-------------------|
| 1. Basic LLM | Getting started | Low | 15 min |
| 2. MCP Agent | Tool integration | Medium | 45 min |
| 3. Multi-Agent | Orchestration | High | 2 hours |
| 4. RAG System | Document QA | Medium | 1 hour |
| 5. Vision Model | Image understanding | Medium | 45 min |
| 6. Interpretability | Model analysis | High | 2 hours |
| 7. Stateful Agent | Persistence | High | 1.5 hours |
| 8. Model Comparison | Benchmarking | Medium | 1 hour |
| 9. Fine-Tuning | Model customization | Very High | 4+ hours |
| 10. Production | Deployment | High | 3 hours |

---

## Next Steps

1. Start with Example 1 to verify your local setup works
2. Progress through examples 2-5 to learn core capabilities
3. Tackle advanced examples 6-7 once comfortable
4. Use examples 8-10 for production applications

All examples are runnable as-is after completing the setup in the main README.
