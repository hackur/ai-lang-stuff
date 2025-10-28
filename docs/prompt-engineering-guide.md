# Prompt Engineering Guide for Local Models

**Last Updated**: 2025-10-26
**Target Audience**: Developers working with Ollama, LM Studio, and local LLMs

---

## Table of Contents

1. [Fundamentals](#fundamentals)
2. [Patterns for Local Models](#patterns-for-local-models)
3. [Model-Specific Tips](#model-specific-tips)
4. [Agent Workflows](#agent-workflows)
5. [RAG Optimization](#rag-optimization)
6. [Common Issues](#common-issues)
7. [Testing & Iteration](#testing--iteration)
8. [Examples Repository](#examples-repository)

---

## Fundamentals

### How Local Models Differ from Cloud Models

Local models have unique characteristics that affect prompting strategies:

| Aspect | Cloud Models (GPT-4, Claude) | Local Models (Qwen3, Gemma3) |
|--------|------------------------------|------------------------------|
| **Context Window** | 128K-200K tokens | 8K-128K tokens (model dependent) |
| **Training Data** | Massive, recent datasets | Smaller, may be older |
| **Instruction Following** | Highly trained for chat | Varies by model |
| **Latency** | Network dependent (200-2000ms) | Local only (50-500ms) |
| **Cost** | Per-token pricing | Zero marginal cost |
| **Privacy** | Data sent to provider | Fully local |
| **Consistency** | Highly consistent | May vary more with temperature |

**Key Implication**: Local models benefit more from explicit, structured prompts with clear examples.

### Token Limits and Context Windows

Understanding context windows is critical for local models:

```python
# Context window sizes for common models
CONTEXT_WINDOWS = {
    "qwen3:8b": 32768,        # 32K tokens
    "qwen3:30b-a3b": 32768,   # 32K tokens
    "gemma3:12b": 8192,       # 8K tokens
    "gemma3:4b": 8192,        # 8K tokens
    "deepseek-r1:8b": 131072, # 128K tokens
    "qwen3-vl:8b": 32768,     # 32K tokens
}

# Rough token estimation
def estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 characters for English)"""
    return len(text) // 4

# Example: Managing context
prompt = "Analyze this code..."
context = read_large_file()  # Could be 100K+ tokens

if estimate_tokens(prompt + context) > CONTEXT_WINDOWS["qwen3:8b"]:
    # Truncate or summarize context
    context = context[:100000]  # Keep first ~25K tokens
```

**Best Practices**:
- Leave 20% buffer for response tokens
- Monitor token usage in production
- Implement context rotation for long conversations
- Use smaller models for simpler tasks to maximize throughput

### Temperature and Sampling Strategies

Temperature controls randomness in model outputs:

```python
from langchain_ollama import ChatOllama

# Temperature guide for different tasks
TEMPERATURE_SETTINGS = {
    "code_generation": 0.1,      # Deterministic, precise
    "data_extraction": 0.0,      # Fully deterministic
    "creative_writing": 0.8,     # High creativity
    "brainstorming": 0.9,        # Maximum diversity
    "general_chat": 0.7,         # Balanced
    "structured_output": 0.2,    # Low variance
    "reasoning": 0.3,            # Focused but not rigid
}

# Example: Code generation
llm_code = ChatOllama(
    model="qwen3:8b",
    temperature=0.1,
    top_p=0.9,          # Nucleus sampling
    repeat_penalty=1.1,  # Reduce repetition
)

# Example: Creative task
llm_creative = ChatOllama(
    model="qwen3:8b",
    temperature=0.8,
    top_p=0.95,
    top_k=40,           # Consider top 40 tokens
)
```

**Advanced Sampling Parameters**:

```python
# Fine-tuned control
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.7,
    top_p=0.9,           # Nucleus sampling threshold
    top_k=40,            # Limit to top K tokens
    repeat_penalty=1.1,  # Penalize repetition (1.0-2.0)
    presence_penalty=0.0, # Penalize token presence
    frequency_penalty=0.0, # Penalize token frequency
)
```

### Stop Sequences

Stop sequences halt generation at specific tokens:

```python
# Common stop sequences
STOP_SEQUENCES = {
    "conversation": ["Human:", "User:", "\n\nHuman"],
    "code": ["```\n\n", "# End of code"],
    "json": ["\n}\n\n", "\n]\n\n"],
    "list": ["\n\n", "---"],
    "qa": ["Question:", "Q:"],
}

# Example: Structured dialogue
llm = ChatOllama(
    model="qwen3:8b",
    stop=["Human:", "\n\nUser:"]
)

response = llm.invoke("""
Assistant: I can help with that. What would you like to know?
Human: Tell me about Python.
Assistant: Python is a high-level programming language...
Human:""")
# Stops before generating more "Human:" prompts
```

---

## Patterns for Local Models

### System Prompts That Work Well

**Pattern 1: Role + Task + Constraints**

```python
# ❌ BAD: Vague system prompt
system_prompt_bad = "You are a helpful assistant."

# ✅ GOOD: Specific role with clear constraints
system_prompt_good = """You are a Python code reviewer specialized in FastAPI applications.

Your responsibilities:
1. Identify security vulnerabilities
2. Suggest performance optimizations
3. Check PEP 8 compliance
4. Recommend better error handling

Constraints:
- Provide specific line numbers for issues
- Explain WHY each change improves the code
- Limit suggestions to 5 most critical items
- Use code snippets to show improvements

Output format:
## Critical Issues
- [Line X] Issue description
  ```python
  # Improved code
  ```

## Suggestions
- [Line Y] Suggestion description
"""
```

**Pattern 2: Persona + Examples + Output Format**

```python
# ✅ EXCELLENT: Complete system prompt
system_prompt_excellent = """You are an expert SQL query optimizer with 10+ years of database experience.

Persona:
- You explain complex concepts in simple terms
- You always consider both performance and readability
- You prefer PostgreSQL-specific features when beneficial

Task:
Analyze SQL queries and suggest optimizations. Focus on:
- Index usage
- Query plan efficiency
- N+1 query problems
- Unnecessary JOINs

Output Format:
### Original Query Analysis
- Estimated cost: [HIGH/MEDIUM/LOW]
- Main bottleneck: [description]

### Optimized Query
```sql
-- Optimized version with comments
```

### Explanation
[Why this is better, expected performance gain]

Example:
User: "SELECT * FROM users WHERE email LIKE '%@gmail.com'"
Assistant:
### Original Query Analysis
- Estimated cost: HIGH
- Main bottleneck: Full table scan due to leading wildcard

### Optimized Query
```sql
-- Option 1: Add functional index
CREATE INDEX idx_users_email_domain ON users ((split_part(email, '@', 2)));
SELECT * FROM users WHERE split_part(email, '@', 2) = 'gmail.com';

-- Option 2: Use trigram index for flexible search
CREATE INDEX idx_users_email_trgm ON users USING gin(email gin_trgm_ops);
SELECT * FROM users WHERE email LIKE '%@gmail.com';
```

### Explanation
Original query can't use indexes due to leading wildcard. Option 1 is fastest for exact domain matches. Option 2 allows flexible pattern matching with ~100x speedup on large tables.
"""
```

### Few-Shot Examples

Few-shot learning dramatically improves local model performance:

```python
# ❌ BAD: No examples
prompt_bad = "Extract the name, email, and phone number from this text."

# ✅ GOOD: 2-3 examples
prompt_good = """Extract name, email, and phone number from text. Return as JSON.

Example 1:
Input: "Contact John Doe at john.doe@example.com or call 555-0123"
Output: {"name": "John Doe", "email": "john.doe@example.com", "phone": "555-0123"}

Example 2:
Input: "Reach out to jane@company.io, phone: (555) 456-7890"
Output: {"name": null, "email": "jane@company.io", "phone": "(555) 456-7890"}

Example 3:
Input: "Meeting with Dr. Sarah Smith tomorrow"
Output: {"name": "Dr. Sarah Smith", "email": null, "phone": null}

Now extract from:
Input: "{user_input}"
Output:"""

# Python implementation
def create_few_shot_prompt(examples: list[dict], task: str) -> str:
    """Generate few-shot prompt from examples."""
    prompt_parts = [f"Task: {task}\n"]

    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Input: {example['input']}")
        prompt_parts.append(f"Output: {example['output']}")

    prompt_parts.append("\nNow complete:")
    prompt_parts.append("Input: {user_input}")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)

# Usage
examples = [
    {"input": "Blue shirt, size M, $29.99", "output": '{"item": "Blue shirt", "size": "M", "price": 29.99}'},
    {"input": "Red sneakers XL $89", "output": '{"item": "Red sneakers", "size": "XL", "price": 89.00}'},
]

prompt = create_few_shot_prompt(examples, "Extract product information as JSON")
```

### Chain-of-Thought Prompting

CoT improves reasoning for complex tasks:

```python
# ❌ BAD: Direct question
prompt_bad = "What is 15% of 234 plus 87?"

# ✅ GOOD: Chain-of-thought
prompt_good = """Solve this step-by-step:

Question: What is 15% of 234 plus 87?

Let's think through this:
1. First, calculate 15% of 234
2. Then add 87 to the result
3. Show the final answer

Solution:
"""

# ✅ EXCELLENT: Few-shot CoT
prompt_excellent = """Solve math problems step-by-step.

Example 1:
Question: What is 20% of 150 plus 30?
Reasoning:
1. Calculate 20% of 150: 150 × 0.20 = 30
2. Add 30: 30 + 30 = 60
Answer: 60

Example 2:
Question: If a product costs $80 and is on 25% sale, what's the final price?
Reasoning:
1. Calculate discount: 80 × 0.25 = 20
2. Subtract from original: 80 - 20 = 60
Answer: $60

Now solve:
Question: What is 15% of 234 plus 87?
Reasoning:
"""

# Python implementation for complex reasoning
class ChainOfThoughtPrompt:
    @staticmethod
    def create(question: str, domain: str = "general") -> str:
        templates = {
            "math": "Let's solve this step-by-step:\n1. Identify the operation needed\n2. Perform calculations\n3. Verify the result\n\nQuestion: {question}\n\nStep-by-step solution:",
            "code": "Let's debug this systematically:\n1. Understand what the code should do\n2. Identify the bug\n3. Explain why it's wrong\n4. Provide the fix\n\nCode: {question}\n\nAnalysis:",
            "logic": "Let's reason through this:\n1. State the given facts\n2. Apply logical rules\n3. Draw conclusions\n4. Verify consistency\n\nProblem: {question}\n\nReasoning:",
        }
        return templates.get(domain, "Think step-by-step:\n{question}\n\nAnalysis:").format(question=question)
```

### Role-Based Prompting

Assigning specific roles improves output quality:

```python
# ❌ BAD: Generic request
prompt_bad = "Review this code: def calc(x, y): return x/y"

# ✅ GOOD: Specific role
prompt_good = """You are a senior Python developer reviewing a pull request.

Review this code for production readiness:
```python
def calc(x, y):
    return x / y
```

Check for:
- Error handling
- Type hints
- Documentation
- Edge cases
"""

# ✅ EXCELLENT: Multi-perspective roles
prompt_excellent = """Review this code from THREE perspectives:

1. **Security Analyst**: Check for vulnerabilities
2. **Performance Engineer**: Identify bottlenecks
3. **Maintainability Expert**: Assess code quality

Code:
```python
def calc(x, y):
    return x / y
```

Provide separate analysis for each role:

## Security Analysis
[Analysis here]

## Performance Analysis
[Analysis here]

## Maintainability Analysis
[Analysis here]

## Overall Recommendation
[Summary and priority fixes]
"""

# Python implementation
ROLE_TEMPLATES = {
    "security_expert": """You are a cybersecurity expert specializing in {language} applications.
Focus on: input validation, injection attacks, authentication, authorization, data exposure.
Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO""",

    "performance_engineer": """You are a performance optimization specialist for {language}.
Focus on: time complexity, space complexity, database queries, caching, concurrency.
Provide: bottleneck identification, optimization suggestions, expected improvements.""",

    "code_reviewer": """You are a senior {language} developer conducting code review.
Focus on: readability, maintainability, design patterns, best practices, testing.
Provide: specific improvements, code examples, rationale for each suggestion.""",
}

def create_role_prompt(role: str, language: str, content: str) -> str:
    """Create a role-based prompt."""
    role_intro = ROLE_TEMPLATES[role].format(language=language)
    return f"{role_intro}\n\n{content}"
```

### Structured Output (JSON Mode)

Getting reliable JSON from local models:

```python
# ❌ BAD: Hoping for JSON
prompt_bad = "Extract entities from: 'Apple released iPhone 15 in September 2023'"

# ✅ GOOD: Explicit JSON schema
prompt_good = """Extract entities and return valid JSON matching this schema:

{
  "organizations": ["string"],
  "products": ["string"],
  "dates": ["string"]
}

Rules:
- Return ONLY valid JSON, no other text
- Use null for missing fields
- Use empty arrays if no entities found

Text: "Apple released iPhone 15 in September 2023"

JSON output:
"""

# ✅ EXCELLENT: JSON mode with validation
from pydantic import BaseModel, Field
from typing import List, Optional

class EntityExtraction(BaseModel):
    """Schema for entity extraction."""
    organizations: List[str] = Field(default_factory=list, description="Company or organization names")
    products: List[str] = Field(default_factory=list, description="Product names")
    dates: List[str] = Field(default_factory=list, description="Date mentions")
    locations: Optional[List[str]] = Field(default=None, description="Geographic locations")

def create_json_prompt(schema: type[BaseModel], text: str) -> str:
    """Create a prompt that enforces JSON schema."""
    schema_json = schema.model_json_schema()

    prompt = f"""Extract structured information from the text and return valid JSON.

Required Schema:
```json
{json.dumps(schema_json, indent=2)}
```

Rules:
1. Return ONLY valid JSON matching the schema exactly
2. Do not include any text before or after the JSON
3. Use null for optional missing fields
4. Use empty arrays for missing list fields
5. Ensure all field types match the schema

Text to analyze:
"{text}"

JSON output:
"""
    return prompt

# Usage with validation
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2, format="json")

prompt = create_json_prompt(EntityExtraction, "Apple released iPhone 15 in September 2023")
response = llm.invoke(prompt)

# Validate against schema
try:
    parsed = EntityExtraction.model_validate_json(response.content)
    print(f"Valid extraction: {parsed}")
except Exception as e:
    print(f"Invalid JSON: {e}")
    # Retry logic here
```

---

## Model-Specific Tips

### Qwen3 Prompting Best Practices

Qwen3 models excel at reasoning and multilingual tasks:

```python
# Qwen3 strengths
QWEN3_STRENGTHS = [
    "Code generation and debugging",
    "Mathematical reasoning",
    "Multilingual understanding (140+ languages)",
    "Long context handling (32K tokens)",
    "Tool/function calling",
]

# ✅ Optimal Qwen3 prompt structure
qwen3_prompt = """<|im_start|>system
You are a helpful AI assistant specialized in {domain}.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
"""

# Example: Code generation with Qwen3
code_gen_prompt = """<|im_start|>system
You are an expert Python developer. Write clean, well-documented code following PEP 8.
<|im_end|>
<|im_start|>user
Create a function that validates email addresses using regex. Include type hints and docstring.
<|im_end|>
<|im_start|>assistant
Here's a robust email validation function:

```python
import re
from typing import Optional

def validate_email(email: str) -> bool:
    \"\"\"
    Validate email address using RFC 5322 regex pattern.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
    \"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```
"""

# Qwen3 excels at multilingual tasks
multilingual_prompt = """<|im_start|>system
You are a multilingual translator. Maintain tone and context across languages.
<|im_end|>
<|im_start|>user
Translate to French, Spanish, and Chinese:
"The quick brown fox jumps over the lazy dog."
<|im_end|>
<|im_start|>assistant
"""

# Qwen3 MoE (30b-a3b) optimization
# Use for parallel tasks and high throughput
moe_prompt = """Process these 5 code snippets in parallel and identify bugs:

1. [Snippet 1]
2. [Snippet 2]
...

For each, provide: bug description, severity, and fix.
"""
```

**Qwen3-Specific Settings**:

```python
# Optimal settings for different Qwen3 tasks
QWEN3_CONFIGS = {
    "code": {
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "stop": ["<|im_end|>", "```\n\n"],
    },
    "reasoning": {
        "temperature": 0.3,
        "top_p": 0.95,
        "repeat_penalty": 1.05,
    },
    "creative": {
        "temperature": 0.8,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
    },
}

llm = ChatOllama(
    model="qwen3:8b",
    **QWEN3_CONFIGS["code"]
)
```

### Gemma3 Specifics

Gemma3 models are optimized for efficiency and safety:

```python
# Gemma3 strengths
GEMMA3_STRENGTHS = [
    "Fast inference (optimized for edge)",
    "Strong safety filters",
    "Excellent multilingual (140+ languages)",
    "Efficient quantization (4-bit performs well)",
    "Good instruction following",
]

# ✅ Gemma3 prompt format
gemma3_prompt = """<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
"""

# Example: Gemma3 for classification tasks
classification_prompt = """<start_of_turn>user
Classify the sentiment of these customer reviews as POSITIVE, NEGATIVE, or NEUTRAL.

Review 1: "Amazing product, exceeded expectations!"
Review 2: "Terrible quality, stopped working after a week."
Review 3: "It's okay, nothing special."

Provide classification for each review:
<end_of_turn>
<start_of_turn>model
Classification results:

1. Review 1: POSITIVE
   Reason: Strong positive language ("amazing", "exceeded expectations")

2. Review 2: NEGATIVE
   Reason: Critical feedback about quality and durability

3. Review 3: NEUTRAL
   Reason: Lukewarm response, neither praise nor criticism
"""

# Gemma3 works well with structured prompts
structured_prompt = """<start_of_turn>user
Extract key information from this job posting:

Title: Senior Python Developer
Location: Remote
Salary: $120,000 - $150,000
Requirements:
- 5+ years Python experience
- FastAPI/Django knowledge
- AWS experience preferred

Format your response as:
## Job Title
[title]

## Details
- Location: [location]
- Salary Range: [range]

## Requirements
- [requirement 1]
- [requirement 2]
...
<end_of_turn>
<start_of_turn>model
"""

# Gemma3 optimal settings
GEMMA3_CONFIGS = {
    "gemma3:12b": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["<end_of_turn>"],
    },
    "gemma3:4b": {  # Lower temp for smaller model
        "temperature": 0.5,
        "top_p": 0.85,
        "top_k": 30,
    },
}
```

### DeepSeek R1 Optimization

DeepSeek R1 specializes in reasoning and mathematics:

```python
# DeepSeek R1 strengths
DEEPSEEK_STRENGTHS = [
    "Advanced mathematical reasoning",
    "Long-form logical analysis",
    "128K context window",
    "Scientific paper comprehension",
    "Multi-step problem solving",
]

# ✅ DeepSeek R1 reasoning prompt
deepseek_prompt = """<|system|>
You are DeepSeek R1, an advanced reasoning model. Break down complex problems into logical steps.
<|user|>
{question}
<|assistant|>
Let me analyze this systematically:

## Problem Understanding
[Restate the problem]

## Approach
[Outline the solution strategy]

## Step-by-Step Solution
1. [First step with explanation]
2. [Second step with explanation]
...

## Verification
[Check the solution]

## Final Answer
[Concise answer]
"""

# Example: Complex reasoning task
math_reasoning = """<|system|>
You are DeepSeek R1, specialized in mathematical proofs and logical reasoning.
<|user|>
Prove that the sum of angles in any triangle is 180 degrees using Euclidean geometry.
<|assistant|>
## Proof: Sum of Angles in a Triangle

**Given**: Triangle ABC with angles α, β, and γ

**To Prove**: α + β + γ = 180°

**Proof**:
1. Draw a line through point A parallel to BC (call it line l)
   - By Euclidean parallel postulate, this line exists and is unique

2. Label the angles formed:
   - Angle on left of A = α₁
   - Angle at A (original) = α
   - Angle on right of A = α₂

3. Apply alternate interior angles theorem:
   - Since l || BC and AB is a transversal: α₁ = β
   - Since l || BC and AC is a transversal: α₂ = γ

4. Observe that angles on line l form a straight angle:
   - α₁ + α + α₂ = 180° (angles on a straight line)

5. Substitute from step 3:
   - β + α + γ = 180°

**Verification**:
- This holds for all triangles in Euclidean geometry
- In non-Euclidean geometry (spherical/hyperbolic), the sum differs

**Q.E.D.**
"""

# DeepSeek R1 optimal settings for reasoning
DEEPSEEK_CONFIGS = {
    "reasoning": {
        "temperature": 0.2,
        "top_p": 0.95,
        "repeat_penalty": 1.05,
        "max_tokens": 4096,  # Allow long reasoning chains
    }
}
```

### Vision Model Prompts (Qwen3-VL)

Qwen3-VL handles multimodal inputs (image + text):

```python
# Qwen3-VL strengths
QWEN3_VL_STRENGTHS = [
    "Document understanding (charts, diagrams)",
    "OCR and text extraction",
    "Visual reasoning",
    "UI/UX analysis",
    "Meme and context understanding",
]

# ✅ Vision prompt structure
vision_prompt = """<|im_start|>system
You are a visual analysis expert. Describe images accurately and answer questions based on visual content.
<|im_end|>
<|im_start|>user
<image>
{text_query}
<|im_end|>
<|im_start|>assistant
"""

# Example: Document analysis
document_analysis = """<|im_start|>system
You are a document analysis expert. Extract structured information from images of documents.
<|im_end|>
<|im_start|>user
<image>
This is a screenshot of an invoice. Extract:
1. Invoice number
2. Date
3. Total amount
4. Line items with quantities and prices

Return as JSON.
<|im_end|>
<|im_start|>assistant
```json
{
  "invoice_number": "INV-2023-001",
  "date": "2023-10-15",
  "total_amount": 1234.56,
  "currency": "USD",
  "line_items": [
    {"description": "Widget A", "quantity": 10, "unit_price": 50.00, "total": 500.00},
    {"description": "Service B", "quantity": 1, "unit_price": 734.56, "total": 734.56}
  ]
}
```
"""

# Example: UI analysis
ui_analysis = """<|im_start|>system
You are a UX designer analyzing user interfaces for accessibility and usability.
<|im_end|>
<|im_start|>user
<image>
Analyze this mobile app screenshot for:
1. Color contrast issues
2. Touch target sizes
3. Information hierarchy
4. Accessibility concerns
<|im_end|>
<|im_start|>assistant
## UI Analysis Report

### Color Contrast
- ❌ Submit button (blue #2E5CFF on white) has ratio 3.2:1 - fails WCAG AA (needs 4.5:1)
- ✅ Body text (dark gray #333 on white) has ratio 12.6:1 - passes AAA

### Touch Target Sizes
- ❌ Social media icons are 24x24px - below recommended 44x44px
- ✅ Primary CTA button is 48x48px - meets guidelines

### Information Hierarchy
- ✅ Clear visual hierarchy with heading, subheading, body text
- ⚠️ Two CTAs compete for attention - consider making secondary button less prominent

### Accessibility Concerns
1. Missing alt text indicators for images
2. Input fields lack visible labels (placeholder-only)
3. No visible focus indicators for keyboard navigation
4. Consider adding skip-to-content link

### Recommendations
1. Increase submit button contrast to #0041C4
2. Enlarge social icons to 44x44px minimum
3. Add proper form labels above inputs
4. Implement focus rings with 2px outline
"""

# Python implementation for vision tasks
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def analyze_image(image_path: str, query: str) -> str:
    """Analyze image using Qwen3-VL."""
    llm = ChatOllama(model="qwen3-vl:8b", temperature=0.3)

    # Encode image as base64
    import base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"},
            {"type": "text", "text": query},
        ]
    )

    response = llm.invoke([message])
    return response.content
```

---

## Agent Workflows

### Prompts for Tool-Using Agents

Enable agents to effectively use tools:

```python
# ✅ EXCELLENT: Tool-using agent prompt
tool_agent_prompt = """You are a data analysis assistant with access to the following tools:

**Available Tools**:
1. **search_database(query: str)**: Search the product database
   - Input: Natural language query
   - Output: List of matching products with details

2. **calculate(expression: str)**: Perform mathematical calculations
   - Input: Mathematical expression (e.g., "15 * 1.08")
   - Output: Numerical result

3. **get_weather(location: str)**: Get current weather
   - Input: City name or coordinates
   - Output: Temperature, conditions, forecast

**Instructions**:
- Use tools when needed to answer questions accurately
- Show your reasoning: explain which tool to use and why
- Format tool calls as: `TOOL: tool_name(arguments)`
- If a tool fails, try an alternative approach
- Combine multiple tools if needed

**Response Format**:
1. **Analysis**: [Understand the question]
2. **Tools Needed**: [List which tools to use]
3. **Execution**: [Make tool calls]
4. **Answer**: [Provide final answer based on tool results]

**Example**:
User: "What's the total cost of 3 blue widgets in Seattle's weather?"

Analysis: Need product pricing and weather information
Tools Needed: search_database, calculate, get_weather

Execution:
TOOL: search_database("blue widgets")
Result: Blue Widget - $29.99 each

TOOL: calculate("3 * 29.99")
Result: 89.97

TOOL: get_weather("Seattle")
Result: 62°F, Partly Cloudy

Answer: The total cost for 3 blue widgets is $89.97. Current weather in Seattle is 62°F and partly cloudy.
"""

# Python implementation with LangChain
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the product database for items matching the query."""
    # Implementation here
    return f"Found products matching '{query}'"

@tool
def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    try:
        return eval(expression, {"__builtins__": {}})
    except Exception as e:
        return f"Error: {e}"

# Create agent
llm = ChatOllama(model="qwen3:8b", temperature=0)
tools = [search_database, calculate]

prompt = ChatPromptTemplate.from_messages([
    ("system", tool_agent_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = executor.invoke({"input": "What's 15% discount on a $99 item?"})
```

### Multi-Agent Communication

Prompts for agents communicating with each other:

```python
# ✅ Multi-agent communication protocol
AGENT_COMMUNICATION_PROMPT = """You are {agent_name}, part of a multi-agent system.

**Your Role**: {role_description}

**Communication Protocol**:
- Address messages using: "@AgentName: message"
- Use structured format for clarity
- Request information: "REQUEST @AgentName: [what you need]"
- Provide information: "RESPONSE @AgentName: [information]"
- Signal completion: "DONE: [summary of your work]"

**Other Agents**:
{other_agents}

**Workflow**:
1. Receive task or message
2. Determine if you can handle it alone
3. If not, request help from appropriate agent
4. Process information from other agents
5. Provide response or signal completion

**Example Interaction**:
@Researcher: REQUEST: Find 3 recent papers on RAG systems
@Researcher: RESPONSE: Found papers [1] ... [2] ... [3] ...
@Summarizer: REQUEST: Summarize these 3 papers
@Summarizer: RESPONSE: Summary: [content]
@Synthesizer: Combining summaries into final report
@Synthesizer: DONE: Report complete with 3 paper summaries
"""

# Example: Research team
researcher_prompt = AGENT_COMMUNICATION_PROMPT.format(
    agent_name="Researcher",
    role_description="Search for and retrieve relevant information from documents and databases",
    other_agents="- Summarizer: Condenses information\n- Validator: Checks accuracy\n- Writer: Creates final output"
)

summarizer_prompt = AGENT_COMMUNICATION_PROMPT.format(
    agent_name="Summarizer",
    role_description="Condense long documents into concise summaries",
    other_agents="- Researcher: Provides source documents\n- Validator: Checks summaries\n- Writer: Uses summaries"
)

# Python implementation with LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    current_agent: str
    task_complete: bool

def researcher_node(state: AgentState) -> AgentState:
    """Researcher agent node."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    prompt = f"""{researcher_prompt}

Current messages:
{chr(10).join(state['messages'])}

Your action:"""

    response = llm.invoke(prompt)

    return {
        "messages": [f"@Researcher: {response.content}"],
        "current_agent": "summarizer",
        "task_complete": "DONE" in response.content,
    }

def summarizer_node(state: AgentState) -> AgentState:
    """Summarizer agent node."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    prompt = f"""{summarizer_prompt}

Current messages:
{chr(10).join(state['messages'])}

Your action:"""

    response = llm.invoke(prompt)

    return {
        "messages": [f"@Summarizer: {response.content}"],
        "current_agent": "end",
        "task_complete": "DONE" in response.content,
    }

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("summarizer", summarizer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "summarizer")
workflow.add_conditional_edges(
    "summarizer",
    lambda s: "end" if s["task_complete"] else "researcher",
    {"end": END, "researcher": "researcher"}
)

app = workflow.compile()
```

### Supervisor Prompts

Supervisor agents coordinate other agents:

```python
# ✅ EXCELLENT: Supervisor agent prompt
supervisor_prompt = """You are a Supervisor Agent coordinating a team of specialized agents.

**Your Responsibilities**:
1. Analyze incoming tasks
2. Delegate to appropriate agents
3. Monitor progress
4. Synthesize results
5. Ensure task completion

**Available Agents**:
- **CodeAgent**: Writes and debugs code
- **ResearchAgent**: Finds information and documentation
- **TestAgent**: Creates and runs tests
- **ReviewAgent**: Reviews code quality

**Decision Process**:
For each task:
1. Break down into subtasks
2. Assign each subtask to the best agent
3. Define success criteria
4. Monitor execution
5. Handle failures (retry, reassign, or escalate)
6. Combine results

**Communication Format**:
TASK_ANALYSIS:
- Main task: [description]
- Subtasks: [list]
- Agents needed: [list]

DELEGATION:
- @AgentName: [specific subtask] | Priority: [HIGH/MEDIUM/LOW] | Deadline: [timeframe]

MONITORING:
- Agent status: [WORKING/BLOCKED/COMPLETE]
- Progress: [percentage or description]

SYNTHESIS:
- Combined results: [summary]
- Quality check: [PASS/FAIL]

**Example**:
User: "Create a FastAPI endpoint for user authentication"

TASK_ANALYSIS:
- Main task: Build authentication endpoint
- Subtasks:
  1. Design API structure
  2. Implement endpoint code
  3. Write unit tests
  4. Review security
- Agents needed: CodeAgent, TestAgent, ReviewAgent

DELEGATION:
- @CodeAgent: Implement POST /auth/login endpoint with JWT | Priority: HIGH | Deadline: 30min
- @TestAgent: Create unit tests for authentication flow | Priority: HIGH | Deadline: 20min
- @ReviewAgent: Security review of authentication code | Priority: HIGH | Deadline: 15min

MONITORING:
- @CodeAgent: COMPLETE (endpoint implemented)
- @TestAgent: WORKING (2/5 tests written)
- @ReviewAgent: WAITING (needs code from CodeAgent)

[After all complete]

SYNTHESIS:
- Combined results: Authentication endpoint implemented with JWT, 5 passing tests, security review passed
- Quality check: PASS
- Deliverable: Ready for deployment
"""

# Python implementation
from typing import Literal, Dict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

class SupervisorState(TypedDict):
    task: str
    subtasks: List[Dict]
    agent_results: Dict[str, str]
    status: str

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor makes delegation decisions."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.2)

    prompt = f"""{supervisor_prompt}

Current Task: {state['task']}

Previous Results:
{json.dumps(state['agent_results'], indent=2)}

Decide next action:
1. Which agent to call next?
2. What specific subtask?
3. Or is the task complete?

Decision:"""

    response = llm.invoke(prompt)

    # Parse supervisor decision
    # (In real implementation, use structured output)

    return state

def agent_executor_node(state: SupervisorState) -> SupervisorState:
    """Execute assigned agent task."""
    # Route to appropriate agent based on supervisor decision
    # Execute subtask
    # Return results
    return state

# Build supervisor workflow
workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("execute_agent", agent_executor_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda s: "end" if s["status"] == "complete" else "execute_agent",
    {"end": END, "execute_agent": "execute_agent"}
)
workflow.add_edge("execute_agent", "supervisor")

supervisor_app = workflow.compile()
```

### Reflection and Self-Correction

Enable agents to improve their outputs:

```python
# ✅ EXCELLENT: Reflection prompt
reflection_prompt = """You are a self-reflective AI agent. After generating a response, critique it and improve it.

**Reflection Process**:
1. **Initial Response**: Generate your first attempt
2. **Self-Critique**: Identify weaknesses, errors, or improvements
3. **Revised Response**: Create an improved version
4. **Verification**: Check if revision addresses all issues

**Critique Dimensions**:
- Accuracy: Are facts correct?
- Completeness: Is anything missing?
- Clarity: Is it easy to understand?
- Efficiency: Can it be more concise?
- Correctness: Are there logical errors?

**Format**:
## Initial Response
[First attempt]

## Self-Critique
**Accuracy**: [Issues found]
**Completeness**: [Gaps identified]
**Clarity**: [Confusing parts]
**Efficiency**: [Redundancies]
**Correctness**: [Logical problems]

## Revised Response
[Improved version addressing all critiques]

## Verification
- [✓] All critiques addressed
- [✓] Quality improved
- [✓] Ready for delivery
"""

# Example: Code generation with reflection
code_reflection_example = """User: "Write a function to find prime numbers up to n"

## Initial Response
```python
def find_primes(n):
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes
```

## Self-Critique
**Accuracy**: ✓ Logic is correct
**Completeness**: ✗ Missing type hints, docstring, edge case handling
**Clarity**: ✗ Variable names could be clearer
**Efficiency**: ✗ Very inefficient - checks all numbers up to num, should use sqrt optimization
**Correctness**: ✗ Should be range(2, n+1) to include n

## Revised Response
```python
def find_primes(n: int) -> list[int]:
    \"\"\"
    Find all prime numbers up to and including n using optimized trial division.

    Args:
        n: Upper bound (inclusive)

    Returns:
        List of prime numbers from 2 to n

    Examples:
        >>> find_primes(10)
        [2, 3, 5, 7]

    Time Complexity: O(n * sqrt(n))
    Space Complexity: O(n)
    \"\"\"
    if n < 2:
        return []

    primes = []

    for candidate in range(2, n + 1):
        is_prime = True

        # Only check divisors up to sqrt(candidate)
        for divisor in range(2, int(candidate ** 0.5) + 1):
            if candidate % divisor == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(candidate)

    return primes
```

## Verification
- [✓] Added type hints
- [✓] Added comprehensive docstring
- [✓] Fixed range to include n
- [✓] Optimized to check only up to sqrt
- [✓] Added edge case handling (n < 2)
- [✓] Clear variable names
- [✓] Included complexity analysis
"""

# Python implementation
def reflect_and_improve(initial_response: str, task: str) -> str:
    """Generate reflection and improved response."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    reflection_step = f"""{reflection_prompt}

Task: {task}

## Initial Response
{initial_response}

Now provide self-critique and revised response:
"""

    reflection = llm.invoke(reflection_step)
    return reflection.content

# Multi-round reflection
def iterative_reflection(task: str, max_iterations: int = 3) -> str:
    """Iteratively improve response through multiple reflection rounds."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    # Generate initial response
    current_response = llm.invoke(f"Task: {task}\n\nResponse:").content

    for i in range(max_iterations):
        # Reflect and improve
        improved = reflect_and_improve(current_response, task)

        # Check if improvement is significant
        # (In real implementation, use quality metrics)

        current_response = improved

    return current_response
```

---

## RAG Optimization

### Query Rewriting

Improve retrieval by rewriting user queries:

```python
# ✅ EXCELLENT: Query rewriting prompt
query_rewrite_prompt = """You are a query optimization specialist for RAG systems.

**Your Task**: Rewrite user queries to improve document retrieval.

**Techniques**:
1. **Expansion**: Add synonyms and related terms
2. **Decomposition**: Break complex queries into subqueries
3. **Clarification**: Make ambiguous queries more specific
4. **Contextualization**: Add domain-specific terminology

**Output Format**:
## Original Query
[User's original query]

## Analysis
- Intent: [What user wants to find]
- Ambiguities: [Unclear parts]
- Missing context: [Implicit assumptions]

## Rewritten Queries
1. **Primary**: [Main optimized query]
2. **Alternative 1**: [Different angle]
3. **Alternative 2**: [Broader/narrower scope]

## Search Strategy
- Recommended: [Which query to use first]
- Fallback: [If primary yields poor results]
"""

# Examples
query_rewrite_examples = """
Example 1:
## Original Query
"How do I fix the login bug?"

## Analysis
- Intent: Find solution for authentication issue
- Ambiguities: Which login system? What kind of bug?
- Missing context: Technology stack, error symptoms

## Rewritten Queries
1. **Primary**: "authentication login error troubleshooting debugging fix"
2. **Alternative 1**: "user authentication failure bug JWT session token"
3. **Alternative 2**: "login endpoint not working 401 403 error"

## Search Strategy
- Recommended: Use primary query first
- Fallback: If no results, try alternative 1 (more technical), then 2 (error-focused)

---

Example 2:
## Original Query
"best practices"

## Analysis
- Intent: Unknown - too vague
- Ambiguities: Best practices for what? Which domain?
- Missing context: Technology, use case, industry

## Rewritten Queries
1. **Primary**: "software development best practices code quality standards"
2. **Alternative 1**: "API design best practices REST GraphQL patterns"
3. **Alternative 2**: "database design best practices normalization indexing"

## Search Strategy
- Recommended: Request clarification from user
- Fallback: Try primary (general), then offer alternatives based on results
"""

# Python implementation
def rewrite_query(original_query: str, context: str = "") -> Dict[str, List[str]]:
    """Rewrite query for better RAG retrieval."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    prompt = f"""{query_rewrite_prompt}

{query_rewrite_examples}

Now rewrite this query:

## Original Query
{original_query}

## Context
{context if context else "No additional context provided"}

Generate analysis and rewritten queries:
"""

    response = llm.invoke(prompt)

    # Parse response (in real implementation, use structured output)
    return {
        "original": original_query,
        "rewritten": [
            "Parsed query 1",
            "Parsed query 2",
            "Parsed query 3",
        ],
        "strategy": "Parsed strategy"
    }

# Multi-query retrieval
def multi_query_retrieval(query: str, retriever, top_k: int = 5) -> List[Dict]:
    """Retrieve documents using multiple query variants."""
    # Rewrite query
    queries = rewrite_query(query)

    all_docs = []
    seen_ids = set()

    # Retrieve with each query variant
    for rewritten_q in queries["rewritten"]:
        docs = retriever.get_relevant_documents(rewritten_q)

        # Deduplicate
        for doc in docs:
            doc_id = hash(doc.page_content)
            if doc_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc_id)

    # Re-rank combined results
    # (In real implementation, use cross-encoder or LLM re-ranking)

    return all_docs[:top_k]
```

### Context Compression

Reduce context size while preserving relevant information:

```python
# ✅ EXCELLENT: Context compression prompt
compression_prompt = """You are a context compression specialist for RAG systems.

**Your Task**: Compress retrieved documents to fit context limits while preserving information relevant to the query.

**Compression Techniques**:
1. **Extraction**: Pull out only relevant sentences
2. **Summarization**: Condense verbose explanations
3. **Deduplication**: Remove repeated information
4. **Prioritization**: Keep most important facts

**Rules**:
- Preserve exact quotes if critical
- Maintain technical accuracy
- Keep numbers, dates, and names exact
- Remove fluff and redundancy
- Aim for 50-70% compression ratio

**Output Format**:
## Original Size
[X tokens]

## Compressed Size
[Y tokens]

## Compression Ratio
[Z%]

## Compressed Content
[Condensed information preserving key facts relevant to query]
"""

# Example
compression_example = """
Query: "How do I implement JWT authentication in FastAPI?"

## Original Content (1000 tokens):
"JSON Web Tokens (JWT) are an open standard (RFC 7519) that defines a compact and self-contained way for securely transmitting information between parties as a JSON object. This information can be verified and trusted because it is digitally signed. JWTs can be signed using a secret (with the HMAC algorithm) or a public/private key pair using RSA or ECDSA.

When implementing JWT authentication in FastAPI, you need to first understand the structure of a JWT. A JWT consists of three parts separated by dots (.), which are: Header, Payload, and Signature. The header typically consists of two parts: the type of the token, which is JWT, and the signing algorithm being used, such as HMAC SHA256 or RSA.

To implement this in FastAPI, you'll want to start by installing the required dependencies. You'll need python-jose for JWT handling and passlib for password hashing. You can install these with pip install python-jose[cryptography] passlib[bcrypt].

Then you'll create a security scheme... [continues for 800 more tokens]"

## Compressed Content (350 tokens):
"JWT authentication in FastAPI implementation steps:

1. **Install dependencies**:
   - python-jose[cryptography] (JWT handling)
   - passlib[bcrypt] (password hashing)

2. **Create security scheme**:
   ```python
   from fastapi.security import OAuth2PasswordBearer
   oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
   ```

3. **Generate JWT token**:
   ```python
   from jose import jwt

   def create_token(data: dict):
       return jwt.encode(data, SECRET_KEY, algorithm="HS256")
   ```

4. **Verify token**:
   ```python
   def verify_token(token: str):
       payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
       return payload
   ```

5. **Protect endpoints**:
   ```python
   @app.get("/protected")
   async def protected(token: str = Depends(oauth2_scheme)):
       user = verify_token(token)
       return user
   ```

Key points: Use HS256 for simplicity, store SECRET_KEY securely, add token expiration, handle decode errors."

## Compression Ratio
65% (1000 → 350 tokens)
"""

# Python implementation
def compress_context(documents: List[str], query: str, max_tokens: int) -> str:
    """Compress retrieved documents to fit within token limit."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.2)

    # Combine documents
    combined = "\n\n---\n\n".join(documents)

    prompt = f"""{compression_prompt}

{compression_example}

Now compress these documents for the query:

Query: {query}

Documents:
{combined}

Target: {max_tokens} tokens maximum

Compressed output:
"""

    compressed = llm.invoke(prompt)
    return compressed.content

# Extractive compression (faster alternative)
def extractive_compression(documents: List[str], query: str, top_sentences: int = 10) -> str:
    """Extract most relevant sentences using embeddings."""
    from sentence_transformers import SentenceTransformer, util
    import nltk

    # Split into sentences
    all_sentences = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc)
        all_sentences.extend(sentences)

    # Embed query and sentences
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(all_sentences, convert_to_tensor=True)

    # Compute similarity
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

    # Get top sentences
    top_indices = similarities.argsort(descending=True)[:top_sentences]
    top_sentences_list = [all_sentences[i] for i in top_indices]

    return " ".join(top_sentences_list)
```

### Citation Formatting

Ensure proper attribution in RAG responses:

```python
# ✅ EXCELLENT: Citation prompt
citation_prompt = """You are a RAG response generator with strict citation requirements.

**Citation Rules**:
1. Every factual claim must cite source document
2. Use inline citations: [1], [2], etc.
3. Include citation list at end
4. Use exact quotes for direct statements
5. Paraphrase with attribution for ideas

**Citation Format**:
[Answer with inline citations]

**Sources**:
[1] Document title or identifier - Relevant excerpt
[2] Document title or identifier - Relevant excerpt

**Guidelines**:
- Multiple sources for same fact: [1,2,3]
- Quote direct text: "exact quote" [1]
- Paraphrase ideas: Summary of concept [1]
- No citation for common knowledge
- Prefer primary sources over secondary
"""

# Example
citation_example = """
Query: "What are the benefits of using FastAPI?"

Retrieved Documents:
Doc1: "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It's one of the fastest Python frameworks available, comparable to NodeJS and Go."

Doc2: "Key features of FastAPI include automatic API documentation with Swagger UI and ReDoc, built-in data validation using Pydantic, and async support for high concurrency."

Doc3: "Developers report 2-3x productivity gains with FastAPI due to reduced boilerplate code and excellent editor support with autocompletion."

Response with Citations:
FastAPI is a high-performance web framework for Python that offers several key benefits [1,2]:

1. **Speed**: It's one of the fastest Python frameworks available, with performance comparable to NodeJS and Go [1]

2. **Automatic Documentation**: FastAPI provides built-in API documentation using Swagger UI and ReDoc [2], eliminating the need for manual API documentation maintenance

3. **Type Safety**: Built-in data validation using Pydantic and Python type hints [2] catches errors early and improves code quality

4. **Developer Productivity**: Developers report "2-3x productivity gains" [3] due to:
   - Reduced boilerplate code [3]
   - Excellent editor support with autocompletion [3]
   - Async support for high concurrency applications [2]

**Sources**:
[1] FastAPI Documentation - "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It's one of the fastest Python frameworks available, comparable to NodeJS and Go."

[2] FastAPI Features Guide - "Key features of FastAPI include automatic API documentation with Swagger UI and ReDoc, built-in data validation using Pydantic, and async support for high concurrency."

[3] Developer Survey 2023 - "Developers report 2-3x productivity gains with FastAPI due to reduced boilerplate code and excellent editor support with autocompletion."
"""

# Python implementation
def generate_with_citations(query: str, documents: List[Dict[str, str]]) -> str:
    """Generate RAG response with proper citations."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    # Format documents with IDs
    doc_context = "\n\n".join([
        f"Doc{i+1}: {doc['content']}"
        for i, doc in enumerate(documents)
    ])

    prompt = f"""{citation_prompt}

{citation_example}

Now answer this query using the provided documents with proper citations:

Query: {query}

Retrieved Documents:
{doc_context}

Response with citations:
"""

    response = llm.invoke(prompt)
    return response.content

# Verify citations
def verify_citations(response: str, documents: List[str]) -> Dict[str, any]:
    """Verify that citations in response are accurate."""
    import re

    # Extract citation numbers
    citations = re.findall(r'\[(\d+(?:,\d+)*)\]', response)

    # Check each citation
    verification = {
        "total_citations": len(citations),
        "missing_sources": [],
        "uncited_claims": [],
    }

    # Extract cited document IDs
    cited_ids = set()
    for citation in citations:
        ids = citation.split(',')
        cited_ids.update(int(id.strip()) for id in ids)

    # Check for missing sources
    for doc_id in cited_ids:
        if doc_id > len(documents):
            verification["missing_sources"].append(doc_id)

    # Advanced: Check for uncited factual claims
    # (Would require NLP analysis in real implementation)

    return verification
```

### Hallucination Reduction

Minimize factual errors in RAG responses:

```python
# ✅ EXCELLENT: Hallucination prevention prompt
hallucination_prevention_prompt = """You are a RAG response generator with STRICT accuracy requirements.

**CRITICAL RULES**:
1. ONLY use information from provided documents
2. If information is not in documents, explicitly state: "This information is not available in the provided sources"
3. NEVER invent facts, statistics, or quotes
4. Distinguish between definite facts and possibilities
5. Use qualifiers: "according to the document", "the source states"

**Verification Process**:
Before including any fact:
1. Can I find this EXACT information in a document?
2. Am I paraphrasing accurately without adding details?
3. Am I making assumptions beyond what's stated?

**Response Structure**:
## Answer
[Information from documents only, with qualifiers]

## Confidence Assessment
- High confidence: [Facts directly stated]
- Medium confidence: [Reasonable inferences]
- Cannot answer: [Questions without source support]

## Information Gaps
[What the user asked that isn't in documents]
"""

# Example
hallucination_example = """
Query: "What are the system requirements and pricing for Product X?"

Retrieved Documents:
Doc1: "Product X requires Python 3.8 or higher and 4GB of RAM."

❌ BAD Response (with hallucinations):
Product X requires Python 3.8+ and 4GB RAM, plus 10GB disk space. It costs $99/month for the basic plan and $299/month for enterprise, with a 14-day free trial available. The software runs on Windows, macOS, and Linux.

❌ Issues:
- Invented disk space requirement (not in docs)
- Invented pricing (not in docs)
- Invented free trial (not in docs)
- Invented OS compatibility (not in docs)

✅ GOOD Response (accurate):
According to the provided documentation, Product X has the following system requirements:
- Python 3.8 or higher [1]
- 4GB of RAM [1]

## Confidence Assessment
- High confidence: Python and RAM requirements are explicitly stated

## Information Gaps
The following information requested was not available in the provided sources:
- Disk space requirements
- Pricing information
- Operating system compatibility
- Free trial availability

To get complete information, please consult the official Product X documentation or contact their sales team.

**Sources**:
[1] Product X Requirements - "Product X requires Python 3.8 or higher and 4GB of RAM."
"""

# Python implementation
def generate_without_hallucination(query: str, documents: List[str]) -> str:
    """Generate response with strict hallucination prevention."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.1)  # Lower temp

    doc_context = "\n\n".join([
        f"Document {i+1}: {doc}"
        for i, doc in enumerate(documents)
    ])

    prompt = f"""{hallucination_prevention_prompt}

{hallucination_example}

IMPORTANT: Only use information from these documents. If information is missing, explicitly state it.

Query: {query}

Available Documents:
{doc_context}

Response:
"""

    response = llm.invoke(prompt)
    return response.content

# Hallucination detection
def detect_hallucinations(response: str, documents: List[str]) -> Dict[str, any]:
    """Detect potential hallucinations in RAG response."""
    llm = ChatOllama(model="qwen3:8b", temperature=0)

    verification_prompt = f"""You are a fact-checking AI. Verify if the response contains information not present in the source documents.

Source Documents:
{chr(10).join(documents)}

Response to Verify:
{response}

For each claim in the response, check:
1. Is this information explicitly in the documents? (YES/NO)
2. If not explicit, is it a reasonable inference? (YES/NO)
3. Or is it potentially hallucinated? (YES/NO)

Output format:
## Verified Claims
- [Claim 1]: SOURCE: [Doc reference]
- [Claim 2]: SOURCE: [Doc reference]

## Potential Hallucinations
- [Claim X]: NOT FOUND in sources
- [Claim Y]: INFERENCE beyond document scope

## Verdict
[SAFE/REVIEW NEEDED/CONTAINS HALLUCINATIONS]
"""

    verification = llm.invoke(verification_prompt)
    return verification.content
```

---

## Common Issues

### Repetition Problems

Local models sometimes generate repetitive output:

```python
# ❌ PROBLEM: Repetitive output
"""
The best way to do this is to use FastAPI. FastAPI is a modern framework.
FastAPI is fast and easy to use. With FastAPI you can build APIs quickly.
FastAPI uses Python type hints. Type hints make FastAPI powerful...
"""

# ✅ SOLUTION 1: Adjust sampling parameters
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.7,
    repeat_penalty=1.2,  # Increase from default 1.1
    top_k=40,
    top_p=0.9,
)

# ✅ SOLUTION 2: Use stop sequences
llm = ChatOllama(
    model="qwen3:8b",
    stop=["\n\n\n", "In summary", "To summarize"],  # Stop before repetitive conclusions
)

# ✅ SOLUTION 3: Explicit anti-repetition prompt
anti_repetition_prompt = """Provide a concise, non-repetitive answer to the question.

IMPORTANT RULES:
- Make each sentence provide NEW information
- Avoid restating the same concept in different words
- If you've made a point, move on to the next point
- Aim for variety in sentence structure
- Be concise - quality over quantity

Question: {question}

Concise answer:
"""

# ✅ SOLUTION 4: Post-processing deduplication
def remove_repetition(text: str, similarity_threshold: float = 0.8) -> str:
    """Remove repetitive sentences from generated text."""
    from sentence_transformers import SentenceTransformer, util
    import nltk

    sentences = nltk.sent_tokenize(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed sentences
    embeddings = model.encode(sentences)

    # Keep first occurrence of similar sentences
    kept_sentences = []
    kept_embeddings = []

    for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
        if i == 0:
            kept_sentences.append(sent)
            kept_embeddings.append(emb)
        else:
            # Check similarity with kept sentences
            similarities = util.cos_sim(emb, kept_embeddings)[0]
            if max(similarities) < similarity_threshold:
                kept_sentences.append(sent)
                kept_embeddings.append(emb)

    return " ".join(kept_sentences)
```

### Following Instructions

Improve instruction adherence:

```python
# ❌ PROBLEM: Model ignores specific instructions
prompt_bad = "Write a Python function. Use type hints."
# Output: def calculate(x, y): return x + y  (no type hints)

# ✅ SOLUTION 1: Emphasize critical requirements
prompt_good = """Write a Python function with these MANDATORY requirements:

MUST HAVE:
1. Type hints for all parameters and return value
2. Docstring with Args, Returns, Examples
3. Error handling for edge cases

MUST NOT:
- Omit any type hints
- Skip the docstring
- Ignore error handling

Function specification: Calculate the sum of two numbers

Output:
"""

# ✅ SOLUTION 2: Use templates with placeholders
template_prompt = """Complete this function template by filling in the implementation:

```python
def calculate_sum(x: int, y: int) -> int:
    \"\"\"
    Calculate the sum of two integers.

    Args:
        x: First integer
        y: Second integer

    Returns:
        Sum of x and y

    Raises:
        TypeError: If inputs are not integers

    Examples:
        >>> calculate_sum(2, 3)
        5
    \"\"\"
    # TODO: Add implementation here
    pass
```

Fill in the implementation:
"""

# ✅ SOLUTION 3: Stepwise instruction following
stepwise_prompt = """Follow these steps EXACTLY in order:

Step 1: Write the function signature with type hints
Step 2: Write the docstring
Step 3: Add error handling
Step 4: Implement the logic
Step 5: Add return statement

Show your work for each step:

STEP 1 - Function signature:
```python
def calculate_sum(x: int, y: int) -> int:
```

STEP 2 - Docstring:
```python
    \"\"\"
    [Your docstring here]
    \"\"\"
```

[Continue for all steps...]
"""

# ✅ SOLUTION 4: Few-shot with correct examples
few_shot_instruction_prompt = """Write functions following this EXACT pattern:

Example 1:
def multiply(x: int, y: int) -> int:
    \"\"\"Multiply two integers.\"\"\"
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("Both arguments must be integers")
    return x * y

Example 2:
def divide(x: float, y: float) -> float:
    \"\"\"Divide x by y.\"\"\"
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("Arguments must be numeric")
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

Now write a function that follows this EXACT same pattern:
Task: Calculate the sum of two numbers
"""

# ✅ SOLUTION 5: Validation in prompt
validation_prompt = """Write a Python function to calculate sum.

After writing, verify it meets these criteria:
✓ Has type hints
✓ Has docstring
✓ Has error handling
✓ Follows PEP 8

Include this checklist at the end:
## Verification
- [✓] Type hints present
- [✓] Docstring complete
- [✓] Error handling added
- [✓] PEP 8 compliant

Function:
"""
```

### Output Formatting

Ensure consistent output format:

```python
# ❌ PROBLEM: Inconsistent formatting
# Sometimes returns JSON, sometimes plain text, sometimes broken JSON

# ✅ SOLUTION 1: Use JSON mode
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2,
    format="json",  # Force JSON output
)

# ✅ SOLUTION 2: Strict format specification with examples
format_prompt = """Return results in this EXACT JSON format (no deviation allowed):

REQUIRED FORMAT:
```json
{
  "status": "success" | "error",
  "data": {
    "field1": "value1",
    "field2": "value2"
  },
  "metadata": {
    "timestamp": "ISO-8601 datetime",
    "model": "model-name"
  }
}
```

EXAMPLE OUTPUT:
```json
{
  "status": "success",
  "data": {
    "result": 42,
    "explanation": "The answer to everything"
  },
  "metadata": {
    "timestamp": "2023-10-26T12:00:00Z",
    "model": "qwen3:8b"
  }
}
```

DO NOT include any text before or after the JSON.
DO NOT use markdown code blocks.
Return ONLY valid JSON.

Task: {task}

JSON output:
"""

# ✅ SOLUTION 3: Pydantic schema enforcement
from pydantic import BaseModel, Field, validator
from datetime import datetime

class APIResponse(BaseModel):
    """Enforced response format."""
    status: Literal["success", "error"]
    data: Dict[str, any]
    metadata: Dict[str, str]

    @validator("metadata")
    def validate_metadata(cls, v):
        required = {"timestamp", "model"}
        if not required.issubset(v.keys()):
            raise ValueError(f"Metadata must include {required}")
        return v

def enforce_format(llm_output: str, schema: type[BaseModel]) -> BaseModel:
    """Parse and validate LLM output against schema."""
    try:
        return schema.model_validate_json(llm_output)
    except Exception as e:
        # Retry with format correction prompt
        correction_prompt = f"""The output was invalid. Fix it to match this schema:

Schema: {schema.model_json_schema()}

Invalid output: {llm_output}

Error: {e}

Corrected valid JSON:
"""
        # Retry logic here
        raise

# ✅ SOLUTION 4: Output parsers
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=APIResponse)

format_instructions = parser.get_format_instructions()

prompt = f"""Answer the question and format output according to these instructions:

{format_instructions}

Question: {question}

Formatted output:
"""

# Parse output
parsed = parser.parse(llm_output)
```

### Context Overflow

Handle context window limits:

```python
# ❌ PROBLEM: Context exceeds model's limit
# Error: "Input too long: 35000 tokens, model max: 32768"

# ✅ SOLUTION 1: Token counting and truncation
def count_tokens(text: str) -> int:
    """Estimate token count (rough: 1 token ≈ 4 chars)."""
    return len(text) // 4

def truncate_to_limit(text: str, max_tokens: int, strategy: str = "end") -> str:
    """Truncate text to fit within token limit."""
    current_tokens = count_tokens(text)

    if current_tokens <= max_tokens:
        return text

    # Calculate character limit
    char_limit = max_tokens * 4

    if strategy == "start":
        return text[:char_limit]
    elif strategy == "end":
        return text[-char_limit:]
    elif strategy == "middle":
        half = char_limit // 2
        return text[:half] + "\n[...truncated...]\n" + text[-half:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ✅ SOLUTION 2: Sliding window for long documents
def sliding_window_process(text: str, window_size: int, overlap: int, llm):
    """Process long text using sliding window with overlap."""
    tokens = count_tokens(text)

    if tokens <= window_size:
        return llm.invoke(text)

    # Split into overlapping windows
    char_window = window_size * 4
    char_overlap = overlap * 4

    windows = []
    start = 0

    while start < len(text):
        end = start + char_window
        windows.append(text[start:end])
        start += char_window - char_overlap

    # Process each window
    results = []
    for window in windows:
        result = llm.invoke(window)
        results.append(result.content)

    # Combine results
    return "\n\n".join(results)

# ✅ SOLUTION 3: Hierarchical summarization
def hierarchical_summarization(long_text: str, llm, chunk_size: int = 20000):
    """Summarize long text hierarchically."""
    # Split into chunks
    chunks = []
    start = 0

    while start < len(long_text):
        end = start + chunk_size
        chunks.append(long_text[start:end])
        start = end

    # First level: Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary_prompt = f"Summarize this text concisely:\n\n{chunk}\n\nSummary:"
        summary = llm.invoke(summary_prompt).content
        summaries.append(summary)

    # Second level: Combine summaries
    combined_summaries = "\n\n".join(summaries)

    # If still too long, recursively summarize
    if count_tokens(combined_summaries) > chunk_size // 4:
        return hierarchical_summarization(combined_summaries, llm, chunk_size)

    # Final summary
    final_prompt = f"Create a comprehensive summary from these section summaries:\n\n{combined_summaries}\n\nFinal summary:"
    return llm.invoke(final_prompt).content

# ✅ SOLUTION 4: Map-Reduce pattern
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def map_reduce_long_text(file_path: str, llm):
    """Process long document using map-reduce."""
    # Load document
    loader = TextLoader(file_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )
    chunks = splitter.split_documents(docs)

    # Map-reduce chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        verbose=True,
    )

    summary = chain.run(chunks)
    return summary

# ✅ SOLUTION 5: Selective content extraction
def extract_relevant_sections(long_text: str, query: str, llm, max_tokens: int):
    """Extract only sections relevant to query."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer, util

    # Split into sections
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    sections = splitter.split_text(long_text)

    # Embed query and sections
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    section_embeddings = model.encode(sections, convert_to_tensor=True)

    # Compute relevance
    similarities = util.cos_sim(query_embedding, section_embeddings)[0]

    # Select top sections that fit in context
    ranked_sections = sorted(
        zip(sections, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    # Accumulate until token limit
    selected = []
    total_tokens = 0

    for section, score in ranked_sections:
        section_tokens = count_tokens(section)
        if total_tokens + section_tokens <= max_tokens:
            selected.append(section)
            total_tokens += section_tokens
        else:
            break

    return "\n\n".join(selected)
```

---

## Testing & Iteration

### Prompt Versioning

Track and manage prompt versions:

```python
# ✅ EXCELLENT: Prompt versioning system
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class PromptVersion:
    """A versioned prompt template."""
    version: str
    template: str
    created_at: datetime
    author: str
    description: str
    parameters: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class PromptRegistry:
    """Registry for managing prompt versions."""

    def __init__(self, storage_path: str = "prompts.json"):
        self.storage_path = storage_path
        self.prompts: Dict[str, List[PromptVersion]] = {}
        self.load()

    def register(self, name: str, version: PromptVersion):
        """Register a new prompt version."""
        if name not in self.prompts:
            self.prompts[name] = []
        self.prompts[name].append(version)
        self.save()

    def get(self, name: str, version: str = "latest") -> PromptVersion:
        """Get a specific prompt version."""
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found")

        versions = self.prompts[name]

        if version == "latest":
            return max(versions, key=lambda v: v.created_at)

        for v in versions:
            if v.version == version:
                return v

        raise KeyError(f"Version '{version}' not found for prompt '{name}'")

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a prompt."""
        if name not in self.prompts:
            return []
        return [v.version for v in self.prompts[name]]

    def save(self):
        """Save registry to disk."""
        data = {}
        for name, versions in self.prompts.items():
            data[name] = [
                {
                    "version": v.version,
                    "template": v.template,
                    "created_at": v.created_at.isoformat(),
                    "author": v.author,
                    "description": v.description,
                    "parameters": v.parameters,
                    "performance_metrics": v.performance_metrics,
                    "tags": v.tags,
                }
                for v in versions
            ]

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load registry from disk."""
        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for name, versions in data.items():
                self.prompts[name] = [
                    PromptVersion(
                        version=v["version"],
                        template=v["template"],
                        created_at=datetime.fromisoformat(v["created_at"]),
                        author=v["author"],
                        description=v["description"],
                        parameters=v.get("parameters", []),
                        performance_metrics=v.get("performance_metrics", {}),
                        tags=v.get("tags", []),
                    )
                    for v in versions
                ]
        except FileNotFoundError:
            pass

# Usage example
registry = PromptRegistry()

# Register v1
registry.register(
    "code_review",
    PromptVersion(
        version="1.0",
        template="Review this code:\n{code}\n\nReview:",
        created_at=datetime.now(),
        author="alice",
        description="Initial code review prompt",
        parameters=["code"],
        tags=["code", "review"],
    )
)

# Register improved v2
registry.register(
    "code_review",
    PromptVersion(
        version="2.0",
        template="""You are a senior code reviewer.

Review this code for:
- Security vulnerabilities
- Performance issues
- Code quality

Code:
{code}

Detailed review:""",
        created_at=datetime.now(),
        author="bob",
        description="Enhanced with specific review criteria",
        parameters=["code"],
        performance_metrics={"accuracy": 0.85, "completeness": 0.90},
        tags=["code", "review", "enhanced"],
    )
)

# Use latest version
latest = registry.get("code_review", "latest")
prompt = latest.template.format(code="def calc(x, y): return x/y")
```

### A/B Testing Locally

Compare prompt performance:

```python
# ✅ EXCELLENT: A/B testing framework
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
import statistics

@dataclass
class ABTestResult:
    """Results from A/B test."""
    variant_a_score: float
    variant_b_score: float
    winner: str
    confidence: float
    sample_size: int
    details: Dict[str, Any]

class PromptABTester:
    """A/B test different prompt variants."""

    def __init__(self, llm, metric_fn: Callable[[str, str], float]):
        """
        Initialize A/B tester.

        Args:
            llm: Language model to test
            metric_fn: Function that scores output (output, expected) -> score
        """
        self.llm = llm
        self.metric_fn = metric_fn

    def run_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: List[Dict[str, str]],
        n_trials: int = 5,
    ) -> ABTestResult:
        """
        Run A/B test on two prompt variants.

        Args:
            prompt_a: First prompt variant
            prompt_b: Second prompt variant
            test_cases: List of {"input": ..., "expected": ...}
            n_trials: Number of trials per test case

        Returns:
            Test results with winner
        """
        scores_a = []
        scores_b = []

        for test_case in test_cases:
            input_text = test_case["input"]
            expected = test_case["expected"]

            # Test prompt A
            for _ in range(n_trials):
                output_a = self.llm.invoke(prompt_a.format(input=input_text))
                score_a = self.metric_fn(output_a.content, expected)
                scores_a.append(score_a)

            # Test prompt B
            for _ in range(n_trials):
                output_b = self.llm.invoke(prompt_b.format(input=input_text))
                score_b = self.metric_fn(output_b.content, expected)
                scores_b.append(score_b)

        # Calculate statistics
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        # Simple winner determination
        winner = "A" if mean_a > mean_b else "B"
        confidence = abs(mean_a - mean_b) / max(mean_a, mean_b)

        return ABTestResult(
            variant_a_score=mean_a,
            variant_b_score=mean_b,
            winner=winner,
            confidence=confidence,
            sample_size=len(scores_a),
            details={
                "scores_a": scores_a,
                "scores_b": scores_b,
                "std_a": statistics.stdev(scores_a),
                "std_b": statistics.stdev(scores_b),
            }
        )

# Example usage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# Define metric function
def accuracy_metric(output: str, expected: str) -> float:
    """Simple accuracy metric (0-1)."""
    return 1.0 if expected.lower() in output.lower() else 0.0

# Define prompts to test
prompt_a = "Classify sentiment: {input}\n\nSentiment:"

prompt_b = """Classify the sentiment of this text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: {input}

Sentiment (one word only):"""

# Test cases
test_cases = [
    {"input": "I love this product!", "expected": "POSITIVE"},
    {"input": "Terrible experience, very disappointed.", "expected": "NEGATIVE"},
    {"input": "It's okay, nothing special.", "expected": "NEUTRAL"},
]

# Run test
tester = PromptABTester(llm, accuracy_metric)
result = tester.run_test(prompt_a, prompt_b, test_cases, n_trials=3)

print(f"Winner: Prompt {result.winner}")
print(f"Scores: A={result.variant_a_score:.2f}, B={result.variant_b_score:.2f}")
print(f"Confidence: {result.confidence:.2%}")
```

### Quality Metrics

Measure prompt effectiveness:

```python
# ✅ EXCELLENT: Quality metrics suite
from typing import List, Dict
from dataclasses import dataclass
import re

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for LLM outputs."""
    accuracy: float  # Correctness (0-1)
    completeness: float  # All required info present (0-1)
    relevance: float  # On-topic (0-1)
    conciseness: float  # Not overly verbose (0-1)
    format_adherence: float  # Follows format requirements (0-1)
    hallucination_rate: float  # Factual errors (0-1, lower better)
    citation_accuracy: float  # Correct citations (0-1)
    overall_score: float  # Weighted average

class QualityEvaluator:
    """Evaluate LLM output quality across multiple dimensions."""

    def __init__(self, llm_judge=None):
        """
        Initialize evaluator.

        Args:
            llm_judge: Optional LLM to use as judge for subjective metrics
        """
        self.llm_judge = llm_judge

    def evaluate(
        self,
        output: str,
        expected: str = None,
        source_docs: List[str] = None,
        format_spec: Dict = None,
    ) -> QualityMetrics:
        """
        Evaluate output quality.

        Args:
            output: LLM output to evaluate
            expected: Expected/reference output
            source_docs: Source documents for RAG
            format_spec: Required format specification

        Returns:
            Quality metrics
        """
        metrics = {}

        # Accuracy (requires expected output)
        if expected:
            metrics["accuracy"] = self._measure_accuracy(output, expected)
        else:
            metrics["accuracy"] = None

        # Completeness
        metrics["completeness"] = self._measure_completeness(output, expected)

        # Relevance (requires LLM judge or expected)
        if self.llm_judge and expected:
            metrics["relevance"] = self._measure_relevance(output, expected)
        else:
            metrics["relevance"] = None

        # Conciseness
        metrics["conciseness"] = self._measure_conciseness(output)

        # Format adherence
        if format_spec:
            metrics["format_adherence"] = self._measure_format_adherence(output, format_spec)
        else:
            metrics["format_adherence"] = None

        # Hallucination rate (requires source docs)
        if source_docs:
            metrics["hallucination_rate"] = self._measure_hallucinations(output, source_docs)
        else:
            metrics["hallucination_rate"] = None

        # Citation accuracy (requires source docs)
        if source_docs:
            metrics["citation_accuracy"] = self._measure_citation_accuracy(output, source_docs)
        else:
            metrics["citation_accuracy"] = None

        # Calculate overall score (weighted average of available metrics)
        weights = {
            "accuracy": 0.3,
            "completeness": 0.2,
            "relevance": 0.15,
            "conciseness": 0.1,
            "format_adherence": 0.1,
            "hallucination_rate": 0.1,
            "citation_accuracy": 0.05,
        }

        available_metrics = {k: v for k, v in metrics.items() if v is not None}
        total_weight = sum(weights[k] for k in available_metrics.keys())

        overall = sum(
            metrics[k] * weights[k] / total_weight
            for k in available_metrics.keys()
            if k != "hallucination_rate"  # Invert for hallucination
        )

        if metrics["hallucination_rate"] is not None:
            overall -= metrics["hallucination_rate"] * weights["hallucination_rate"] / total_weight

        metrics["overall_score"] = max(0, min(1, overall))

        return QualityMetrics(**metrics)

    def _measure_accuracy(self, output: str, expected: str) -> float:
        """Measure factual accuracy against expected output."""
        # Simple overlap-based accuracy
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 1.0

        overlap = len(output_words & expected_words)
        return overlap / len(expected_words)

    def _measure_completeness(self, output: str, expected: str = None) -> float:
        """Measure if all required information is present."""
        if not expected:
            # Heuristic: Check if output is substantial
            return min(1.0, len(output) / 500)  # 500 chars = complete

        # Check if key points from expected are in output
        expected_sentences = expected.split('.')
        matches = sum(1 for sent in expected_sentences if sent.strip().lower() in output.lower())

        return matches / len(expected_sentences) if expected_sentences else 0.0

    def _measure_relevance(self, output: str, expected: str) -> float:
        """Measure relevance to the query (using LLM judge)."""
        if not self.llm_judge:
            return None

        judge_prompt = f"""Rate the relevance of the output to the expected answer on a scale of 0-10.

Expected: {expected}

Output: {output}

Relevance score (0-10):"""

        response = self.llm_judge.invoke(judge_prompt).content

        # Extract score
        match = re.search(r'\d+', response)
        if match:
            score = int(match.group())
            return min(10, max(0, score)) / 10

        return 0.5  # Default if can't parse

    def _measure_conciseness(self, output: str) -> float:
        """Measure conciseness (not overly verbose)."""
        # Ideal length around 200-500 words
        word_count = len(output.split())

        if word_count < 50:
            return 0.5  # Too brief
        elif 50 <= word_count <= 300:
            return 1.0  # Ideal
        elif 300 < word_count <= 500:
            return 0.8  # Acceptable
        else:
            return max(0, 1 - (word_count - 500) / 1000)  # Penalty for verbosity

    def _measure_format_adherence(self, output: str, format_spec: Dict) -> float:
        """Measure adherence to format requirements."""
        score = 1.0

        # Check required fields
        if "required_fields" in format_spec:
            for field in format_spec["required_fields"]:
                if field not in output:
                    score -= 0.2

        # Check format type (e.g., JSON)
        if format_spec.get("type") == "json":
            try:
                import json
                json.loads(output)
            except:
                score -= 0.5

        # Check structure markers
        if "structure" in format_spec:
            for marker in format_spec["structure"]:
                if marker not in output:
                    score -= 0.1

        return max(0, score)

    def _measure_hallucinations(self, output: str, source_docs: List[str]) -> float:
        """Estimate hallucination rate (facts not in sources)."""
        # Extract factual claims from output
        # (Simplified: look for sentences with numbers, dates, names)
        factual_pattern = r'[A-Z][^.!?]*(?:\d+|[A-Z][a-z]+\s+[A-Z][a-z]+)[^.!?]*[.!?]'
        claims = re.findall(factual_pattern, output)

        if not claims:
            return 0.0  # No factual claims

        # Check how many claims are supported by sources
        sources_text = " ".join(source_docs).lower()
        unsupported = 0

        for claim in claims:
            # Simple check: is claim text in sources?
            if claim.lower() not in sources_text:
                unsupported += 1

        return unsupported / len(claims)

    def _measure_citation_accuracy(self, output: str, source_docs: List[str]) -> float:
        """Measure accuracy of citations."""
        # Extract citations
        citations = re.findall(r'\[(\d+)\]', output)

        if not citations:
            return 1.0  # No citations to check

        # Check if cited doc IDs are valid
        max_doc_id = len(source_docs)
        invalid = sum(1 for cit in citations if int(cit) > max_doc_id)

        return 1 - (invalid / len(citations))

# Example usage
evaluator = QualityEvaluator(llm_judge=ChatOllama(model="qwen3:8b"))

output = """FastAPI is a modern web framework for Python [1].
It provides automatic API documentation and supports async operations [1].
FastAPI is 10x faster than Django [2]."""  # [2] is hallucinated

source_docs = [
    "FastAPI is a modern web framework for Python. It provides automatic API documentation and supports async operations."
]

expected = "FastAPI is a Python web framework with automatic documentation and async support."

metrics = evaluator.evaluate(
    output=output,
    expected=expected,
    source_docs=source_docs,
)

print(f"Overall Score: {metrics.overall_score:.2f}")
print(f"Accuracy: {metrics.accuracy:.2f}")
print(f"Hallucination Rate: {metrics.hallucination_rate:.2f}")
print(f"Citation Accuracy: {metrics.citation_accuracy:.2f}")
```

### Benchmark Prompts

Standard test cases for comparing prompts:

```python
# ✅ EXCELLENT: Benchmark suite
from typing import List, Dict, Callable
from dataclasses import dataclass
import json

@dataclass
class BenchmarkCase:
    """A benchmark test case."""
    id: str
    category: str
    input: str
    expected_output: str
    evaluation_fn: Callable[[str, str], float]
    difficulty: str  # "easy", "medium", "hard"
    tags: List[str]

class PromptBenchmark:
    """Benchmark suite for evaluating prompts."""

    def __init__(self):
        self.cases: List[BenchmarkCase] = []
        self._load_standard_benchmarks()

    def _load_standard_benchmarks(self):
        """Load standard benchmark cases."""

        # Code generation benchmarks
        self.cases.append(BenchmarkCase(
            id="code_gen_001",
            category="code_generation",
            input="Write a function to check if a string is a palindrome",
            expected_output="def is_palindrome(s: str) -> bool:",
            evaluation_fn=lambda out, exp: 1.0 if "def is_palindrome" in out else 0.0,
            difficulty="easy",
            tags=["python", "string", "algorithm"],
        ))

        # Data extraction benchmarks
        self.cases.append(BenchmarkCase(
            id="extract_001",
            category="data_extraction",
            input="Extract email from: Contact us at support@example.com for help",
            expected_output="support@example.com",
            evaluation_fn=lambda out, exp: 1.0 if exp in out else 0.0,
            difficulty="easy",
            tags=["extraction", "email"],
        ))

        # Reasoning benchmarks
        self.cases.append(BenchmarkCase(
            id="reason_001",
            category="reasoning",
            input="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            expected_output="No, we cannot conclude that",
            evaluation_fn=lambda out, exp: 1.0 if "cannot" in out.lower() or "no" in out.lower()[:20] else 0.0,
            difficulty="medium",
            tags=["logic", "reasoning"],
        ))

        # Summarization benchmarks
        self.cases.append(BenchmarkCase(
            id="summ_001",
            category="summarization",
            input="""The quick brown fox jumps over the lazy dog. This pangram contains
every letter of the English alphabet at least once. It is commonly used for testing
typewriters and computer keyboards, displaying fonts, and other applications.""",
            expected_output="pangram",
            evaluation_fn=lambda out, exp: 1.0 if exp in out.lower() else 0.0,
            difficulty="easy",
            tags=["summarization", "comprehension"],
        ))

        # More benchmarks...

    def add_case(self, case: BenchmarkCase):
        """Add a custom benchmark case."""
        self.cases.append(case)

    def run_benchmark(self, prompt_template: str, llm, categories: List[str] = None) -> Dict:
        """
        Run benchmark on a prompt template.

        Args:
            prompt_template: Prompt template with {input} placeholder
            llm: Language model to test
            categories: Specific categories to test (None = all)

        Returns:
            Benchmark results
        """
        results = {
            "total_cases": 0,
            "passed": 0,
            "failed": 0,
            "accuracy": 0.0,
            "by_category": {},
            "by_difficulty": {},
            "details": [],
        }

        # Filter cases
        test_cases = self.cases
        if categories:
            test_cases = [c for c in self.cases if c.category in categories]

        # Run each case
        for case in test_cases:
            prompt = prompt_template.format(input=case.input)
            output = llm.invoke(prompt).content

            # Evaluate
            score = case.evaluation_fn(output, case.expected_output)
            passed = score >= 0.8

            # Update results
            results["total_cases"] += 1
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

            # By category
            if case.category not in results["by_category"]:
                results["by_category"][case.category] = {"total": 0, "passed": 0}
            results["by_category"][case.category]["total"] += 1
            if passed:
                results["by_category"][case.category]["passed"] += 1

            # By difficulty
            if case.difficulty not in results["by_difficulty"]:
                results["by_difficulty"][case.difficulty] = {"total": 0, "passed": 0}
            results["by_difficulty"][case.difficulty]["total"] += 1
            if passed:
                results["by_difficulty"][case.difficulty]["passed"] += 1

            # Details
            results["details"].append({
                "id": case.id,
                "category": case.category,
                "difficulty": case.difficulty,
                "passed": passed,
                "score": score,
                "output": output[:200],  # First 200 chars
            })

        # Calculate accuracy
        results["accuracy"] = results["passed"] / results["total_cases"] if results["total_cases"] > 0 else 0.0

        # Calculate category accuracies
        for category, stats in results["by_category"].items():
            stats["accuracy"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0

        # Calculate difficulty accuracies
        for difficulty, stats in results["by_difficulty"].items():
            stats["accuracy"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0

        return results

    def save_results(self, results: Dict, filepath: str):
        """Save benchmark results to file."""
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

    def compare_prompts(self, prompts: Dict[str, str], llm) -> Dict:
        """
        Compare multiple prompt variants.

        Args:
            prompts: Dict of {name: template}
            llm: Language model to test

        Returns:
            Comparison results
        """
        comparison = {}

        for name, template in prompts.items():
            print(f"Testing prompt: {name}")
            results = self.run_benchmark(template, llm)
            comparison[name] = results

        # Determine winner
        best = max(comparison.items(), key=lambda x: x[1]["accuracy"])

        return {
            "results": comparison,
            "winner": best[0],
            "winner_accuracy": best[1]["accuracy"],
        }

# Example usage
benchmark = PromptBenchmark()

# Define prompts to compare
prompts = {
    "basic": "{input}\n\nAnswer:",

    "structured": """Task: {input}

Please provide a clear, concise answer.

Answer:""",

    "cot": """Task: {input}

Let's think step-by-step:
1. Understand the task
2. Plan the approach
3. Execute

Answer:""",
}

# Run comparison
llm = ChatOllama(model="qwen3:8b", temperature=0.3)
comparison = benchmark.compare_prompts(prompts, llm)

print(f"\nWinner: {comparison['winner']}")
print(f"Accuracy: {comparison['winner_accuracy']:.1%}")

# Save results
benchmark.save_results(comparison, "benchmark_results.json")
```

---

## Examples Repository

### Example 1: Zero-Shot vs Few-Shot Classification

```python
# Zero-shot
zero_shot = """Classify the sentiment: "This movie was amazing!"

Sentiment:"""
# Output: "Positive" (works, but inconsistent format)

# Few-shot (better)
few_shot = """Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.

Example: "I love this product!" → POSITIVE
Example: "Terrible service" → NEGATIVE
Example: "It's okay" → NEUTRAL

Classify: "This movie was amazing!" → """
# Output: "POSITIVE" (consistent format)
```

### Example 2: Vague vs Specific Instructions

```python
# Vague
vague = "Write some code for a login system"
# Output: Unpredictable, may skip important features

# Specific (better)
specific = """Write a Python login function with these requirements:

1. Function signature: authenticate(username: str, password: str) -> bool
2. Hash passwords with bcrypt
3. Raise ValueError for empty inputs
4. Return True if authenticated, False otherwise
5. Include docstring with examples

Code:"""
# Output: Meets all requirements
```

### Example 3: Direct Answer vs Chain-of-Thought

```python
# Direct
direct = "What is 15% of 240 plus 30?"
# Output: May skip steps, potential errors

# Chain-of-thought (better)
cot = """Solve step-by-step:

Question: What is 15% of 240 plus 30?

Step 1: Calculate 15% of 240
Step 2: Add 30 to the result
Step 3: State the final answer

Solution:"""
# Output: Shows reasoning, more accurate
```

### Example 4: Unstructured vs Structured Output

```python
# Unstructured
unstructured = "Extract the name, email, and phone from this text"
# Output: Inconsistent format

# Structured (better)
structured = """Extract information as JSON:

{"name": "string or null", "email": "string or null", "phone": "string or null"}

Text: {text}

JSON:"""
# Output: Consistent, parseable
```

### Example 5: Simple vs Role-Based Prompt

```python
# Simple
simple = "Review this code for bugs"
# Output: Generic feedback

# Role-based (better)
role_based = """You are a security-focused code reviewer with 10 years experience.

Review this code for:
1. Security vulnerabilities (SQL injection, XSS, etc.)
2. Input validation issues
3. Authentication/authorization flaws

Prioritize by severity (CRITICAL, HIGH, MEDIUM, LOW).

Code:
{code}

Security review:"""
# Output: Security-focused, actionable
```

---

## Conclusion

This guide covers comprehensive prompt engineering techniques for local models. Key takeaways:

1. **Be Explicit**: Local models need clear, detailed instructions
2. **Use Examples**: Few-shot learning significantly improves performance
3. **Structure Output**: Enforce consistent formats with schemas
4. **Test Iteratively**: Use benchmarks and A/B testing
5. **Model-Specific**: Optimize for each model's strengths
6. **Manage Context**: Stay within token limits with compression
7. **Prevent Hallucinations**: Use strict verification prompts
8. **Version Prompts**: Track changes and performance

For more examples and updates, see the `/Volumes/JS-DEV/ai-lang-stuff/examples/` directory.

---

**Related Resources**:
- [LangChain Prompt Engineering Guide](https://python.langchain.com/docs/modules/model_io/prompts/)
- [OpenAI Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering Tutorial](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Local Model Comparison](../plans/1-research-plan.md)

**Contributing**:
To add examples or improve this guide, submit a PR with:
- New prompt pattern with before/after comparison
- Benchmark results showing improvement
- Code examples that run locally
