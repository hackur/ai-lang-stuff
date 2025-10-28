# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting key technical decisions for the ai-lang-stuff project.

## What are ADRs?

Architecture Decision Records capture important architectural decisions along with their context and consequences. They serve as a historical record of why certain choices were made, helping future contributors understand the project's technical direction.

## ADR Index

### Infrastructure & Platform

- [ADR-001: Local-First Architecture](001-local-first-architecture.md)
  - **Status**: Accepted
  - **Decision**: Implement local-first architecture with optional cloud integration
  - **Key Points**: Privacy-first, cost-effective, offline-capable, runs entirely on-device
  - **Impact**: Foundation for all subsequent decisions

- [ADR-006: Apple Silicon Optimization](006-apple-silicon-optimization.md)
  - **Status**: Accepted
  - **Decision**: Comprehensive optimization for M1/M2/M3 chips using MLX and Metal
  - **Key Points**: 3-6x performance improvement, quantization-first, unified memory
  - **Impact**: Competitive local inference performance

### Frameworks & Orchestration

- [ADR-002: LangGraph for Agent Orchestration](002-langgraph-choice.md)
  - **Status**: Accepted
  - **Decision**: LangGraph as primary multi-agent orchestration framework
  - **Key Points**: Explicit state management, graph-based workflows, checkpointing
  - **Impact**: Deterministic, debuggable agent architectures

- [ADR-003: Model Context Protocol (MCP)](003-mcp-protocol.md)
  - **Status**: Accepted
  - **Decision**: MCP as standard for tool integration
  - **Key Points**: Vendor-neutral protocol, sandboxed execution, standardization
  - **Impact**: Secure, reusable tool integrations across frameworks

### Data & Storage

- [ADR-004: Vector Store Selection](004-vector-store-selection.md)
  - **Status**: Accepted
  - **Decision**: Dual strategy with Chroma (primary) and FAISS (performance)
  - **Key Points**: Ease of use vs performance, both local-first, LangChain integration
  - **Impact**: RAG systems from prototypes to production

- [ADR-005: State Management Strategy](005-state-management.md)
  - **Status**: Accepted
  - **Decision**: SQLite via LangGraph's SqliteSaver for all state persistence
  - **Key Points**: Local-first, ACID compliant, time-travel debugging, thread isolation
  - **Impact**: Reliable state management with zero infrastructure

## Reading Guide

### For New Contributors

Start with these ADRs to understand the project's foundation:
1. ADR-001: Local-First Architecture - understand the "why"
2. ADR-002: LangGraph Choice - grasp the orchestration approach
3. ADR-006: Apple Silicon Optimization - learn the performance strategy

### For Feature Development

Depending on what you're building:
- **Multi-agent workflows**: ADR-002 (LangGraph), ADR-005 (State Management)
- **Tool integration**: ADR-003 (MCP Protocol)
- **RAG systems**: ADR-004 (Vector Stores), ADR-005 (State Management)
- **Performance optimization**: ADR-006 (Apple Silicon)

### For Architecture Review

Review all ADRs to understand:
- Trade-offs made in key decisions
- Alternatives that were considered
- Consequences (both positive and negative)
- Migration paths and implementation strategies

## ADR Format

Each ADR follows this structure:

```markdown
# ADR NNN: Title

## Status
[Accepted | Deprecated | Superseded by ADR-XXX]

## Context
[Problem statement, requirements, stakeholders]

## Decision
[What we decided to do]

## Rationale
[Why we made this decision]

## Consequences
[Positive and negative outcomes]

## Alternatives Considered
[Other options and why they were rejected]

## Implementation
[How to implement this decision]

## Verification
[How to test and validate]

## References
[Links to documentation, discussions, etc.]

## Related ADRs
[Cross-references to other ADRs]

## Changelog
[Version history]
```

## Decision Relationships

```
ADR-001 (Local-First)
    ├── ADR-002 (LangGraph) - orchestration for local models
    ├── ADR-003 (MCP) - local tool execution
    ├── ADR-004 (Vector Stores) - local RAG
    ├── ADR-005 (State Management) - local persistence
    └── ADR-006 (Apple Silicon) - local performance

ADR-002 (LangGraph)
    ├── ADR-005 (State Management) - checkpoint implementation
    └── ADR-003 (MCP) - tool integration layer

ADR-004 (Vector Stores)
    └── ADR-006 (Apple Silicon) - embedding performance
```

## Proposing New ADRs

When proposing a new ADR:

1. **Check existing ADRs** - ensure the decision isn't already covered
2. **Use the template** - follow the standard format above
3. **Number sequentially** - next ADR is 007
4. **Include context** - explain the problem thoroughly
5. **Compare alternatives** - show you've considered options
6. **Document trade-offs** - be honest about consequences
7. **Link related ADRs** - show how decisions interconnect

### ADR Numbering

- 001-099: Core architecture and infrastructure
- 100-199: Framework and library choices
- 200-299: Data and storage strategies
- 300-399: Security and privacy
- 400-499: Developer experience and tooling
- 500+: Future expansion

## Updating ADRs

ADRs are immutable once accepted, but can be:
- **Deprecated**: Mark as deprecated with reason
- **Superseded**: Link to new ADR that replaces it
- **Amended**: Add changelog entry for clarifications only

Never change the core decision after acceptance. Instead, create a new ADR.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on proposing and reviewing ADRs.

## Questions?

- Review related ADRs for context
- Check implementation examples in `examples/`
- Consult [CLAUDE.md](../../CLAUDE.md) for development guidelines
- Ask in discussions or issues

## External Resources

- [ADR Overview](https://adr.github.io/)
- [When to Write an ADR](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR Tools](https://github.com/npryce/adr-tools)
