"""
Code Review Workflow for LangGraph Studio.

Pipeline with conditional routing: Syntax → Security → Style → Approval Gate
Routes to Code Fixer if issues found, then re-reviews.

This workflow demonstrates conditional edges and cyclic flows.
"""

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# State Definition
# ============================================================================


class CodeIssue(TypedDict):
    """Single code issue."""

    severity: Literal["critical", "major", "minor"]
    category: str
    description: str


class CodeReviewState(TypedDict):
    """State for code review pipeline.

    Attributes:
        code: Source code to review
        language: Programming language
        issues: List of issues found
        security_score: Security rating (0-100)
        style_score: Style rating (0-100)
        approved: Whether code is approved
        needs_rewrite: Needs major rewrite
        fixed_code: Code with fixes
        messages: Message history
        iteration: Current iteration
    """

    code: str
    language: str
    issues: Annotated[List[CodeIssue], operator.add]
    security_score: int
    style_score: int
    approved: bool
    needs_rewrite: bool
    fixed_code: str
    messages: Annotated[List, operator.add]
    iteration: int


# ============================================================================
# Review Nodes
# ============================================================================


def syntax_checker_node(state: CodeReviewState) -> CodeReviewState:
    """Check syntax and logic."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.3)

    code_to_check = state.get("fixed_code") or state["code"]

    system = SystemMessage(
        content="You are a syntax checker. Find syntax errors and logical flaws."
    )
    user = HumanMessage(
        content=f"""Check {state['language']} code:

```{state['language']}
{code_to_check}
```

List issues with severity. Respond "SYNTAX_OK" if no issues."""
    )

    response = llm.invoke([system, user])
    issues = []

    if "SYNTAX_OK" not in response.content:
        issues.append(
            {
                "severity": "major",
                "category": "syntax",
                "description": "Syntax issues detected",
            }
        )

    return {
        "issues": issues,
        "messages": [AIMessage(content=f"Found {len(issues)} syntax issues", name="Syntax")],
        "iteration": state.get("iteration", 0) + 1,
    }


def security_scanner_node(state: CodeReviewState) -> CodeReviewState:
    """Scan for security vulnerabilities."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.2)

    code_to_check = state.get("fixed_code") or state["code"]

    system = SystemMessage(
        content="You are a security expert. Find vulnerabilities and rate security 0-100."
    )
    user = HumanMessage(
        content=f"""Security scan {state['language']} code:

```{state['language']}
{code_to_check}
```

Format: SECURITY_SCORE: <number>
Then list vulnerabilities."""
    )

    response = llm.invoke([system, user])

    # Parse score
    security_score = 50
    if "SECURITY_SCORE:" in response.content:
        try:
            score_line = [
                line for line in response.content.split("\n") if "SECURITY_SCORE:" in line
            ][0]
            security_score = int("".join(filter(str.isdigit, score_line))[:2])
        except:
            pass

    issues = []
    if security_score < 70:
        issues.append(
            {
                "severity": "critical" if security_score < 50 else "major",
                "category": "security",
                "description": f"Security issues (score: {security_score})",
            }
        )

    return {
        "security_score": security_score,
        "issues": issues,
        "messages": [
            AIMessage(content=f"Security score: {security_score}/100", name="Security")
        ],
        "iteration": state.get("iteration", 0) + 1,
    }


def style_reviewer_node(state: CodeReviewState) -> CodeReviewState:
    """Review code style."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.4)

    code_to_check = state.get("fixed_code") or state["code"]

    system = SystemMessage(
        content="You are a style expert. Review code style and rate 0-100."
    )
    user = HumanMessage(
        content=f"""Style review {state['language']} code:

```{state['language']}
{code_to_check}
```

Format: STYLE_SCORE: <number>
Then list suggestions."""
    )

    response = llm.invoke([system, user])

    # Parse score
    style_score = 70
    if "STYLE_SCORE:" in response.content:
        try:
            score_line = [
                line for line in response.content.split("\n") if "STYLE_SCORE:" in line
            ][0]
            style_score = int("".join(filter(str.isdigit, score_line))[:2])
        except:
            pass

    issues = []
    if style_score < 60:
        issues.append(
            {
                "severity": "minor",
                "category": "style",
                "description": f"Style improvements needed (score: {style_score})",
            }
        )

    return {
        "style_score": style_score,
        "issues": issues,
        "messages": [AIMessage(content=f"Style score: {style_score}/100", name="Style")],
        "iteration": state.get("iteration", 0) + 1,
    }


def approval_gate_node(state: CodeReviewState) -> CodeReviewState:
    """Make approval decision."""
    critical = sum(1 for i in state.get("issues", []) if i["severity"] == "critical")
    major = sum(1 for i in state.get("issues", []) if i["severity"] == "major")

    security = state.get("security_score", 0)

    if critical > 0:
        decision = "REJECTED - Critical issues"
        approved, needs_rewrite = False, True
    elif major > 2 or security < 60:
        decision = "REJECTED - Too many issues"
        approved, needs_rewrite = False, True
    elif major > 0 or security < 80:
        decision = "NEEDS WORK"
        approved, needs_rewrite = False, False
    else:
        decision = "APPROVED"
        approved, needs_rewrite = True, False

    return {
        "approved": approved,
        "needs_rewrite": needs_rewrite,
        "messages": [AIMessage(content=decision, name="Approval")],
        "iteration": state.get("iteration", 0) + 1,
    }


def code_fixer_node(state: CodeReviewState) -> CodeReviewState:
    """Attempt automatic fixes."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.2)

    issues_summary = "\n".join(
        [f"- [{i['severity']}] {i['category']}: {i['description']}"
         for i in state.get("issues", [])]
    )

    system = SystemMessage(
        content="Fix code issues. Output ONLY corrected code."
    )
    user = HumanMessage(
        content=f"""Fix {state['language']} code:

Original:
```{state['language']}
{state['code']}
```

Issues:
{issues_summary}

Provide fixed code:"""
    )

    response = llm.invoke([system, user])
    fixed = response.content

    # Extract from markdown
    if "```" in fixed:
        fixed = fixed.split("```")[1]
        if "\n" in fixed:
            fixed = "\n".join(fixed.split("\n")[1:])

    return {
        "fixed_code": fixed,
        "issues": [],  # Clear issues for re-review
        "messages": [AIMessage(content="Fixes applied", name="Fixer")],
        "iteration": state.get("iteration", 0) + 1,
    }


# ============================================================================
# Conditional Routing
# ============================================================================


def route_after_style(state: CodeReviewState) -> Literal["approval_gate", "code_fixer"]:
    """Route based on findings."""
    critical = any(i["severity"] == "critical" for i in state.get("issues", []))

    if critical and state.get("iteration", 0) < 3:
        return "code_fixer"
    return "approval_gate"


def route_after_approval(state: CodeReviewState) -> Literal["end", "code_fixer"]:
    """Route after approval."""
    if state.get("needs_rewrite") and state.get("iteration", 0) < 3:
        return "code_fixer"
    return "end"


# ============================================================================
# Graph Construction
# ============================================================================


def create_graph() -> StateGraph:
    """Create code review graph with conditional routing."""
    workflow = StateGraph(CodeReviewState)

    # Add nodes
    workflow.add_node("syntax_checker", syntax_checker_node)
    workflow.add_node("security_scanner", security_scanner_node)
    workflow.add_node("style_reviewer", style_reviewer_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("code_fixer", code_fixer_node)

    # Sequential review
    workflow.set_entry_point("syntax_checker")
    workflow.add_edge("syntax_checker", "security_scanner")
    workflow.add_edge("security_scanner", "style_reviewer")

    # Conditional after style
    workflow.add_conditional_edges(
        "style_reviewer",
        route_after_style,
        {"approval_gate": "approval_gate", "code_fixer": "code_fixer"},
    )

    # Fixer loops back
    workflow.add_edge("code_fixer", "syntax_checker")

    # Conditional after approval
    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {"code_fixer": "code_fixer", "end": END},
    )

    return workflow


# ============================================================================
# LangGraph Studio Entry Point
# ============================================================================

code_reviewer = create_graph().compile(
    checkpointer=SqliteSaver.from_conn_string("./checkpoints/code_review.db")
)

# For standalone testing
if __name__ == "__main__":
    sample_code = '''def process(user_input):
    import os
    os.system(user_input)  # Command injection
    password = "admin123"  # Hardcoded secret
    query = "SELECT * FROM users WHERE id = " + user_input  # SQL injection
    return query'''

    initial_state = {
        "code": sample_code,
        "language": "python",
        "issues": [],
        "security_score": 0,
        "style_score": 0,
        "approved": False,
        "needs_rewrite": False,
        "fixed_code": "",
        "messages": [],
        "iteration": 0,
    }

    config = {"configurable": {"thread_id": "test-001"}}

    print("Code Review Workflow")
    print("=" * 70)

    for step_output in code_reviewer.stream(initial_state, config):
        node_name = list(step_output.keys())[0]
        node_state = step_output[node_name]
        print(f"\n[Iteration {node_state.get('iteration')}] {node_name.upper()}")

    final_state = code_reviewer.get_state(config).values
    print("\n" + "=" * 70)
    print(f"APPROVED: {final_state.get('approved')}")
    print(f"Security: {final_state.get('security_score')}/100")
    print(f"Style: {final_state.get('style_score')}/100")
