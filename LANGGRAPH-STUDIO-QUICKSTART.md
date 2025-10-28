# LangGraph Studio - Quick Start

Visual workflow editing and debugging for LangGraph pipelines.

---

## Start Studio (One Command)

```bash
npm run studio
# Or: npx langgraph@latest dev
```

**Opens**: http://localhost:8123/studio

---

## Available Workflows

| Workflow | Description | Use Case |
|----------|-------------|----------|
| **research_agent** | Sequential research pipeline | Automated research |
| **code_reviewer** | Code review with conditional routing | Code quality checks |
| **rag_pipeline** | RAG with document retrieval | Question answering |

---

## Quick Test

### 1. Research Agent

**Input**:
```json
{
  "question": "What are the benefits of local LLMs?",
  "research_findings": "",
  "analysis": "",
  "summary": "",
  "messages": [],
  "iteration": 0
}
```

**Flow**: Question → Research → Analysis → Summary

---

### 2. Code Reviewer

**Input**:
```json
{
  "code": "def unsafe(input):\n    exec(input)",
  "language": "python",
  "issues": [],
  "security_score": 0,
  "style_score": 0,
  "approved": false,
  "needs_rewrite": false,
  "fixed_code": "",
  "messages": [],
  "iteration": 0
}
```

**Flow**: Syntax → Security → Style → Approval → [Fix or End]

---

### 3. RAG Pipeline

**Input**:
```json
{
  "query": "What are local LLMs?",
  "documents": ["Local LLMs run entirely on your device..."],
  "retrieved_docs": [],
  "context": "",
  "response": "",
  "sources": [],
  "messages": [],
  "iteration": 0
}
```

**Flow**: Ingest → Retrieve → Generate

---

## Standalone Testing

Test workflows without Studio:

```bash
# Research agent
python workflows/research_agent.py

# Code reviewer
python workflows/code_reviewer.py

# RAG pipeline
python workflows/rag_pipeline.py
```

---

## Prerequisites

```bash
# 1. Ollama running
ollama serve

# 2. Model installed
ollama pull qwen3:8b

# 3. Node.js 18+
node --version

# 4. Python 3.10+
python --version
```

---

## Troubleshooting

### Studio won't start
```bash
# Verify config exists
cat langgraph.json

# Check port availability
lsof -i :8123
```

### Ollama connection error
```bash
# Start Ollama
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### Workflow not loading
```bash
# Test standalone
python workflows/research_agent.py

# Check export name
grep "research_agent =" workflows/research_agent.py
```

---

## File Structure

```
ai-lang-stuff/
├── langgraph.json          # Studio configuration
├── workflows/              # Workflow definitions
│   ├── research_agent.py   # Sequential pipeline
│   ├── code_reviewer.py    # Conditional routing
│   └── rag_pipeline.py     # RAG system
├── checkpoints/            # Workflow state persistence
├── .env.example            # Environment template
└── docs/
    └── langgraph-studio-guide.md  # Detailed guide
```

---

## Key Features

- **Visual Debugging**: See workflow execution in real-time
- **State Inspection**: Check state at every checkpoint
- **Breakpoints**: Pause execution at any node
- **Replay**: Re-run from any checkpoint
- **Multi-workflow**: Switch between workflows easily

---

## Next Steps

1. **Start Studio**: `npm run studio`
2. **Run Examples**: Test all 3 workflows
3. **Read Guide**: `docs/langgraph-studio-guide.md`
4. **Create Workflow**: See `workflows/README.md`
5. **Deploy**: Use PostgreSQL checkpointer for production

---

## Resources

- **Full Guide**: `/docs/langgraph-studio-guide.md`
- **Workflow README**: `/workflows/README.md`
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/studio/
- **Discord**: https://discord.gg/langchain

---

**Happy Visual Debugging!**
