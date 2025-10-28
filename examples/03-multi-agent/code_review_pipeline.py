"""
Code Review Pipeline with Conditional Routing using LangGraph.

This example demonstrates a sophisticated code review workflow with conditional
routing based on code quality. The pipeline routes code through different paths
depending on review results.

Architecture:
- State: Tracks code, issues, suggestions, test results, and approval status
- Nodes: Syntax checker → Security scanner → Style reviewer → Decision gate
- Conditional Edges: Route based on severity of issues found
- Persistence: SQLite checkpointer maintains review history

Workflow Paths:
1. Critical issues → Security fixer → Re-review
2. Minor issues → Style suggestions → Optional approval
3. No issues → Approval → END

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`

Expected output:
Complete code review with conditional routing based on findings.
"""

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

# Import utilities
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.state_manager import StateManager, create_thread_id
from utils.logging import setup_logging, get_logger


# ============================================================================
# State Definition
# ============================================================================


class CodeIssue(TypedDict):
    """Represents a single code issue."""

    severity: Literal["critical", "major", "minor", "info"]
    category: str  # "security", "syntax", "style", "performance"
    line: int
    description: str
    suggestion: str


class CodeReviewState(TypedDict):
    """State for code review pipeline.

    Attributes:
        code: Source code being reviewed
        language: Programming language
        issues: List of issues found
        suggestions: List of improvement suggestions
        security_score: Security rating (0-100)
        style_score: Style rating (0-100)
        approved: Whether code is approved
        needs_rewrite: Whether code needs major rewrite
        fixed_code: Code with fixes applied (optional)
        messages: Message history
        iteration: Current review iteration
        errors: List of any errors encountered
    """

    code: str
    language: str
    issues: Annotated[List[CodeIssue], operator.add]
    suggestions: Annotated[List[str], operator.add]
    security_score: int
    style_score: int
    approved: bool
    needs_rewrite: bool
    fixed_code: str
    messages: Annotated[List, operator.add]
    iteration: int
    errors: Annotated[List[str], operator.add]


# ============================================================================
# Review Node Definitions
# ============================================================================


def syntax_checker_node(state: CodeReviewState) -> CodeReviewState:
    """Check code for syntax and logical errors.

    Args:
        state: Current review state

    Returns:
        Updated state with syntax issues
    """
    logger = get_logger(__name__)
    logger.info("SYNTAX CHECKER: Starting syntax analysis")

    try:
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.3,
        )

        system_prompt = SystemMessage(
            content="""You are a syntax checker. Analyze code for syntax errors,
            logical flaws, and basic correctness. Report issues with line numbers
            and specific descriptions. Be thorough but focus on correctness, not style."""
        )

        user_prompt = HumanMessage(
            content=f"""Analyze this {state['language']} code for syntax and logical errors:

```{state['language']}
{state['code']}
```

For each issue found, provide:
- Line number
- Severity (critical/major/minor)
- Description
- Suggestion for fix

If no issues, respond with "SYNTAX_OK"."""
        )

        response = llm.invoke([system_prompt, user_prompt])
        analysis = response.content

        logger.info(f"SYNTAX CHECKER: Analysis complete - {len(analysis)} chars")

        # Parse response for issues (simplified parsing)
        issues = []
        if "SYNTAX_OK" not in analysis:
            # Found issues - create issue objects
            # In production, use structured output or JSON
            issues.append(
                {
                    "severity": "major",
                    "category": "syntax",
                    "line": 0,
                    "description": "Syntax issues detected",
                    "suggestion": analysis[:200],
                }
            )

        return {
            "issues": issues,
            "messages": [
                AIMessage(
                    content=f"Syntax check complete. Found {len(issues)} issues",
                    name="SyntaxChecker",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"SYNTAX CHECKER: Error - {str(e)}")
        return {
            "errors": [f"Syntax checker error: {str(e)}"],
            "messages": [
                AIMessage(
                    content=f"Syntax check failed: {str(e)}",
                    name="SyntaxChecker",
                )
            ],
        }


def security_scanner_node(state: CodeReviewState) -> CodeReviewState:
    """Scan code for security vulnerabilities.

    Args:
        state: Current review state

    Returns:
        Updated state with security issues and score
    """
    logger = get_logger(__name__)
    logger.info("SECURITY SCANNER: Starting security analysis")

    try:
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.2,  # Very low temp for security
        )

        system_prompt = SystemMessage(
            content="""You are a security expert. Analyze code for security
            vulnerabilities including: SQL injection, XSS, CSRF, hardcoded secrets,
            insecure algorithms, path traversal, command injection, etc.
            Rate security on 0-100 scale (100 = perfect security)."""
        )

        user_prompt = HumanMessage(
            content=f"""Perform security analysis on this {state['language']} code:

```{state['language']}
{state['code']}
```

Provide:
1. Security score (0-100)
2. List of vulnerabilities with severity
3. Recommendations for fixes

Format: SECURITY_SCORE: <number>
Then list issues."""
        )

        response = llm.invoke([system_prompt, user_prompt])
        analysis = response.content

        # Parse security score (simplified)
        security_score = 50  # Default
        if "SECURITY_SCORE:" in analysis:
            try:
                score_line = [
                    line
                    for line in analysis.split("\n")
                    if "SECURITY_SCORE:" in line
                ][0]
                security_score = int(
                    "".join(filter(str.isdigit, score_line))[:2]
                )
            except (IndexError, ValueError):
                pass

        # Create issues for critical security problems
        issues = []
        if security_score < 70:
            issues.append(
                {
                    "severity": "critical" if security_score < 50 else "major",
                    "category": "security",
                    "line": 0,
                    "description": f"Security concerns detected (score: {security_score})",
                    "suggestion": analysis[:200],
                }
            )

        logger.info(f"SECURITY SCANNER: Score {security_score}/100")

        return {
            "security_score": security_score,
            "issues": issues,
            "messages": [
                AIMessage(
                    content=f"Security scan complete. Score: {security_score}/100",
                    name="SecurityScanner",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"SECURITY SCANNER: Error - {str(e)}")
        return {
            "errors": [f"Security scanner error: {str(e)}"],
            "messages": [
                AIMessage(
                    content=f"Security scan failed: {str(e)}",
                    name="SecurityScanner",
                )
            ],
        }


def style_reviewer_node(state: CodeReviewState) -> CodeReviewState:
    """Review code style and best practices.

    Args:
        state: Current review state

    Returns:
        Updated state with style suggestions and score
    """
    logger = get_logger(__name__)
    logger.info("STYLE REVIEWER: Starting style review")

    try:
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.4,
        )

        system_prompt = SystemMessage(
            content="""You are a code style expert. Review code for:
            - Code organization and structure
            - Naming conventions
            - Comments and documentation
            - Best practices for the language
            - Readability and maintainability
            Rate style on 0-100 scale (100 = perfect style)."""
        )

        user_prompt = HumanMessage(
            content=f"""Review the style of this {state['language']} code:

```{state['language']}
{state['code']}
```

Provide:
1. Style score (0-100)
2. Specific style suggestions
3. Best practice recommendations

Format: STYLE_SCORE: <number>
Then list suggestions."""
        )

        response = llm.invoke([system_prompt, user_prompt])
        analysis = response.content

        # Parse style score (simplified)
        style_score = 70  # Default
        if "STYLE_SCORE:" in analysis:
            try:
                score_line = [
                    line for line in analysis.split("\n") if "STYLE_SCORE:" in line
                ][0]
                style_score = int("".join(filter(str.isdigit, score_line))[:2])
            except (IndexError, ValueError):
                pass

        # Extract suggestions (simplified)
        suggestions = [analysis[:150] + "..."]

        # Create minor issues for style problems
        issues = []
        if style_score < 60:
            issues.append(
                {
                    "severity": "minor",
                    "category": "style",
                    "line": 0,
                    "description": f"Style improvements needed (score: {style_score})",
                    "suggestion": analysis[:200],
                }
            )

        logger.info(f"STYLE REVIEWER: Score {style_score}/100")

        return {
            "style_score": style_score,
            "issues": issues,
            "suggestions": suggestions,
            "messages": [
                AIMessage(
                    content=f"Style review complete. Score: {style_score}/100",
                    name="StyleReviewer",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"STYLE REVIEWER: Error - {str(e)}")
        return {
            "errors": [f"Style reviewer error: {str(e)}"],
            "messages": [
                AIMessage(
                    content=f"Style review failed: {str(e)}",
                    name="StyleReviewer",
                )
            ],
        }


def approval_gate_node(state: CodeReviewState) -> CodeReviewState:
    """Make final approval decision based on all reviews.

    Args:
        state: Current review state with all issues

    Returns:
        Updated state with approval decision
    """
    logger = get_logger(__name__)
    logger.info("APPROVAL GATE: Making final decision")

    # Count issues by severity
    critical_count = sum(
        1 for issue in state.get("issues", []) if issue["severity"] == "critical"
    )
    major_count = sum(
        1 for issue in state.get("issues", []) if issue["severity"] == "major"
    )

    security_score = state.get("security_score", 0)
    style_score = state.get("style_score", 0)

    # Decision logic
    approved = False
    needs_rewrite = False

    if critical_count > 0:
        needs_rewrite = True
        decision = "REJECTED - Critical issues found"
    elif major_count > 2 or security_score < 60:
        needs_rewrite = True
        decision = "REJECTED - Too many major issues or poor security"
    elif major_count > 0 or security_score < 80:
        approved = False
        decision = "NEEDS WORK - Address issues before approval"
    else:
        approved = True
        decision = "APPROVED - Code meets quality standards"

    logger.info(f"APPROVAL GATE: {decision}")

    return {
        "approved": approved,
        "needs_rewrite": needs_rewrite,
        "messages": [
            AIMessage(
                content=decision,
                name="ApprovalGate",
            )
        ],
        "iteration": state.get("iteration", 0) + 1,
    }


def code_fixer_node(state: CodeReviewState) -> CodeReviewState:
    """Attempt to automatically fix issues found in review.

    Args:
        state: Current review state with issues

    Returns:
        Updated state with fixed code
    """
    logger = get_logger(__name__)
    logger.info("CODE FIXER: Attempting automatic fixes")

    try:
        llm = ChatOllama(
            model="qwen3:8b",
            base_url="http://localhost:11434",
            temperature=0.2,
        )

        # Build fix prompt with all issues
        issues_summary = "\n".join(
            [
                f"- [{issue['severity']}] {issue['category']}: {issue['description']}"
                for issue in state.get("issues", [])
            ]
        )

        system_prompt = SystemMessage(
            content="""You are a code fixing assistant. Given code and a list
            of issues, produce corrected code that addresses all problems.
            Output ONLY the fixed code, no explanations."""
        )

        user_prompt = HumanMessage(
            content=f"""Fix the following {state['language']} code:

Original code:
```{state['language']}
{state['code']}
```

Issues to fix:
{issues_summary}

Provide the corrected code:"""
        )

        response = llm.invoke([system_prompt, user_prompt])
        fixed_code = response.content

        # Extract code from markdown if present
        if "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1]
            if "\n" in fixed_code:
                fixed_code = "\n".join(fixed_code.split("\n")[1:])

        logger.info(f"CODE FIXER: Generated {len(fixed_code)} chars of fixed code")

        return {
            "fixed_code": fixed_code,
            "messages": [
                AIMessage(
                    content=f"Fixes applied. Review again.",
                    name="CodeFixer",
                )
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        logger.error(f"CODE FIXER: Error - {str(e)}")
        return {
            "errors": [f"Code fixer error: {str(e)}"],
            "messages": [
                AIMessage(
                    content=f"Fixing failed: {str(e)}",
                    name="CodeFixer",
                )
            ],
        }


# ============================================================================
# Conditional Routing
# ============================================================================


def route_after_review(
    state: CodeReviewState,
) -> Literal["approval_gate", "code_fixer"]:
    """Route based on review findings.

    Args:
        state: Current state after reviews

    Returns:
        Next node to execute
    """
    logger = get_logger(__name__)

    # Check for critical issues
    critical_issues = [
        issue for issue in state.get("issues", []) if issue["severity"] == "critical"
    ]

    if critical_issues and state.get("iteration", 0) < 2:
        logger.info("ROUTER: Critical issues found, routing to code_fixer")
        return "code_fixer"
    else:
        logger.info("ROUTER: Routing to approval_gate")
        return "approval_gate"


def route_after_approval(state: CodeReviewState) -> Literal["end", "code_fixer"]:
    """Route after approval decision.

    Args:
        state: Current state after approval gate

    Returns:
        Next node or END
    """
    logger = get_logger(__name__)

    if state.get("needs_rewrite") and state.get("iteration", 0) < 2:
        logger.info("ROUTER: Needs rewrite, routing to code_fixer")
        return "code_fixer"
    else:
        logger.info("ROUTER: Review complete, ending")
        return "end"


# ============================================================================
# Graph Construction
# ============================================================================


def create_code_review_pipeline() -> StateGraph:
    """Create the code review pipeline graph with conditional routing.

    Graph structure:
        START → Syntax → Security → Style → [Conditional]

        If critical issues:
            → Code Fixer → Syntax (re-review)

        Else:
            → Approval Gate → [Conditional]

            If needs rewrite:
                → Code Fixer → Syntax (re-review)

            Else:
                → END

    Returns:
        StateGraph: Compiled LangGraph workflow
    """
    logger = get_logger(__name__)
    logger.info("Creating code review pipeline graph")

    # Create graph
    workflow = StateGraph(CodeReviewState)

    # Add nodes
    workflow.add_node("syntax_checker", syntax_checker_node)
    workflow.add_node("security_scanner", security_scanner_node)
    workflow.add_node("style_reviewer", style_reviewer_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("code_fixer", code_fixer_node)

    # Define edges
    workflow.set_entry_point("syntax_checker")
    workflow.add_edge("syntax_checker", "security_scanner")
    workflow.add_edge("security_scanner", "style_reviewer")

    # Conditional routing after reviews
    workflow.add_conditional_edges(
        "style_reviewer",
        route_after_review,
        {
            "approval_gate": "approval_gate",
            "code_fixer": "code_fixer",
        },
    )

    # Code fixer loops back to syntax checker for re-review
    workflow.add_edge("code_fixer", "syntax_checker")

    # Conditional routing after approval
    workflow.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {
            "code_fixer": "code_fixer",
            "end": END,
        },
    )

    logger.info("Code review pipeline graph created successfully")
    return workflow


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run the code review pipeline example."""
    setup_logging()
    logger = get_logger(__name__)

    print("=" * 70)
    print("Code Review Pipeline with Conditional Routing")
    print("=" * 70)
    print()

    try:
        # Create pipeline
        workflow = create_code_review_pipeline()

        # Setup persistence
        checkpointer = StateManager.get_checkpointer(
            "./checkpoints_code_review.db"
        )
        app = workflow.compile(checkpointer=checkpointer)

        # Create thread ID
        thread_id = create_thread_id("code-review")
        config = {"configurable": {"thread_id": thread_id}}

        print(f"Thread ID: {thread_id}")
        print()

        # Sample code to review (intentionally has issues)
        code_sample = '''def process_user_input(user_input):
    # Execute user command
    import os
    os.system(user_input)

    # SQL query without parameterization
    query = "SELECT * FROM users WHERE id = " + user_input

    password = "admin123"  # Hardcoded password

    return query'''

        print("Code to review:")
        print("-" * 70)
        print(code_sample)
        print("-" * 70)
        print()

        # Initialize state
        initial_state = {
            "code": code_sample,
            "language": "python",
            "issues": [],
            "suggestions": [],
            "security_score": 0,
            "style_score": 0,
            "approved": False,
            "needs_rewrite": False,
            "fixed_code": "",
            "messages": [],
            "iteration": 0,
            "errors": [],
        }

        # Execute pipeline
        print("Executing Code Review Pipeline...")
        print("-" * 70)

        for step_output in app.stream(initial_state, config):
            node_name = list(step_output.keys())[0]
            node_state = step_output[node_name]

            print(f"\n[Iteration {node_state.get('iteration', '?')}] {node_name.upper()}")

            # Show progress
            if node_name == "syntax_checker":
                print("  Checking syntax and logic...")
            elif node_name == "security_scanner":
                score = node_state.get("security_score", 0)
                print(f"  Security score: {score}/100")
            elif node_name == "style_reviewer":
                score = node_state.get("style_score", 0)
                print(f"  Style score: {score}/100")
            elif node_name == "approval_gate":
                print(
                    f"  Approved: {node_state.get('approved', False)}, "
                    f"Needs rewrite: {node_state.get('needs_rewrite', False)}"
                )
            elif node_name == "code_fixer":
                print("  Applying automatic fixes...")

        # Get final state
        print("\n" + "=" * 70)
        print("REVIEW RESULTS")
        print("=" * 70)

        final_state = app.get_state(config).values

        print(f"\nTotal iterations: {final_state.get('iteration', 0)}")
        print(f"Security score: {final_state.get('security_score', 0)}/100")
        print(f"Style score: {final_state.get('style_score', 0)}/100")
        print(f"Approved: {final_state.get('approved', False)}")
        print(f"Needs rewrite: {final_state.get('needs_rewrite', False)}")

        # Show issues
        issues = final_state.get("issues", [])
        if issues:
            print(f"\nIssues found: {len(issues)}")
            print("-" * 70)
            for i, issue in enumerate(issues, 1):
                print(
                    f"{i}. [{issue['severity'].upper()}] {issue['category']}: "
                    f"{issue['description']}"
                )

        # Show suggestions
        suggestions = final_state.get("suggestions", [])
        if suggestions:
            print(f"\nSuggestions: {len(suggestions)}")
            print("-" * 70)
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")

        # Show fixed code if available
        if final_state.get("fixed_code"):
            print("\nFixed code:")
            print("-" * 70)
            print(final_state["fixed_code"][:500])
            print("-" * 70)

        # Show errors if any
        if final_state.get("errors"):
            print("\nErrors encountered:")
            for error in final_state["errors"]:
                print(f"  - {error}")

        # Show checkpoint stats
        stats = StateManager.get_checkpoint_size("./checkpoints_code_review.db")
        print(f"\nCheckpoint stats:")
        print(f"  - Database size: {stats['file_size_mb']:.2f} MB")
        print(f"  - Total checkpoints: {stats['checkpoint_count']}")

        print("\n" + "=" * 70)
        print("Code review pipeline complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Code review pipeline failed: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Is the model installed? Try: ollama pull qwen3:8b")
        print("3. Check logs for detailed error information")


if __name__ == "__main__":
    main()
