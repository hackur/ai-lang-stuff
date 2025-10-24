# Skill: Debug Agent

## Purpose
Systematically debug LangChain agents and LangGraph workflows when they fail or produce unexpected results.

## Triggers
- Agent not responding or hanging
- Agent produces incorrect output
- Agent fails with error
- User reports "agent not working"
- Tool calls failing

## Process

### 1. Verify Environment
```bash
# Check Ollama is running
ps aux | grep ollama
ollama list

# Check model availability
ollama list | grep <model-name>

# Test model directly
ollama run <model-name> "test prompt"
```

### 2. Enable Tracing
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_VERBOSE"] = "true"
```

### 3. Check Agent Configuration
- Verify tools are properly defined
- Check prompt template includes {agent_scratchpad}
- Ensure model supports tool calling
- Validate state schema (for LangGraph)

### 4. Test Components Individually
```python
# Test LLM
llm = ChatOllama(model="qwen3:8b")
response = llm.invoke("test")
print(response)

# Test each tool
for tool in tools:
    try:
        result = tool.func("test input")
        print(f"{tool.name}: {result}")
    except Exception as e:
        print(f"{tool.name} ERROR: {e}")

# Test prompt
prompt = agent.get_prompts()[0]
print(prompt.format(input="test", agent_scratchpad=""))
```

### 5. Common Issues & Solutions

#### Issue: "Model not responding"
**Solution**:
- Check Ollama logs: `ollama logs`
- Verify model supports tool calling
- Try simpler prompt
- Increase timeout

#### Issue: "Tool not being called"
**Solution**:
- Check tool description is clear
- Verify tool name doesn't have special characters
- Ensure prompt includes tool usage instructions
- Try with different model

#### Issue: "State not persisting"
**Solution**:
- Verify checkpointer is configured
- Check SQLite database exists
- Ensure thread_id is consistent
- Validate state schema matches

#### Issue: "Infinite loop"
**Solution**:
- Add max_iterations to AgentExecutor
- Check conditional edges don't create cycle
- Add explicit END conditions
- Log state transitions to identify loop

### 6. Add Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add custom logging
def log_state(state):
    print(f"Current state: {state}")
    return state

workflow.add_node("logger", log_state)
```

### 7. Simplify and Rebuild
If still not working:
1. Create minimal working example
2. Add components one at a time
3. Test after each addition
4. Identify breaking component

## Output
- Clear diagnosis of issue
- Step-by-step solution
- Fixed code
- Explanation of what went wrong
- Prevention tips for future

## Example Interaction

**User**: "My agent keeps hanging and not responding"

**Claude**:
"Let me help debug this systematically.

First, let's verify your environment:
```bash
# Check if Ollama is running
ps aux | grep ollama
```

[User confirms Ollama is running]

Now let's test the model directly:
```bash
ollama run qwen3:8b 'test'
```

[Model responds]

Good, the model works. Let's enable tracing and see what the agent is doing:

```python
import os
os.environ['LANGCHAIN_VERBOSE'] = 'true'

# Run your agent
result = agent_executor.invoke({...})
```

[Examines trace]

I see the issue - the agent is waiting for tool output but the tool is timing out. The problem is in your `search_tool` - it's not handling network timeouts.

Here's the fix:
```python
import httpx
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def search_with_timeout(query: str) -> str:
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"http://...", params={"q": query})
        return response.text
```

This adds:
1. 10-second timeout
2. Retry logic (3 attempts)
3. Proper exception handling

Try again and let me know if it works!"
