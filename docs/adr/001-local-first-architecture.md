# ADR 001: Local-First Architecture

## Status
Accepted

## Context
We need to establish the foundational architecture for an AI experimentation toolkit. The choice between local-first, cloud-first, or hybrid approaches has significant implications for privacy, cost, latency, and user experience.

### Problem Statement
- Users need powerful AI capabilities without sacrificing privacy
- Cloud API costs can be prohibitive for experimentation and development
- Network latency affects real-time interactions
- Dependency on cloud services creates availability risks
- Developers want full control over model selection and behavior

### Stakeholders
- Individual developers and researchers
- Teams working with sensitive data
- Users in regions with limited internet connectivity
- Cost-conscious experimenters
- Privacy-focused organizations

## Decision
We will implement a **local-first architecture** with optional cloud integration, where:
1. All core functionality runs entirely on-device using Ollama/LM Studio
2. Cloud services (OpenAI, Anthropic) are available as optional extensions
3. No data leaves the device unless explicitly configured by the user
4. All tools and integrations prioritize local execution

## Rationale

### Why Local-First

**Privacy Benefits:**
- Complete data sovereignty - no telemetry or data collection
- GDPR/HIPAA compliant by default
- Suitable for confidential business data
- No prompt logging or model training on user data
- Offline-capable for sensitive environments

**Cost Analysis:**
- Zero ongoing API costs for base functionality
- One-time hardware investment (M3 Max: ~$3,500)
- Break-even vs cloud after ~350-700 hours of heavy usage
- Unlimited experimentation without metering
- No surprise bills from high-volume testing

**Performance Characteristics:**
- 20-50 tokens/second on M3 Max (qwen3:8b)
- Zero network latency (<1ms local inference start)
- Consistent performance regardless of internet quality
- Predictable resource utilization

**Developer Experience:**
- Full model transparency and control
- Ability to modify models and prompts freely
- Reproducible environments
- Works on airplanes, remote locations
- No API rate limits or quotas

### Tradeoffs

**Advantages:**
- ✅ Complete privacy and data control
- ✅ Predictable costs (hardware only)
- ✅ Low latency for inference
- ✅ Offline capability
- ✅ No rate limits
- ✅ Full transparency into model behavior
- ✅ Suitable for sensitive data

**Disadvantages:**
- ❌ Requires capable hardware (16GB+ RAM, preferably Apple Silicon)
- ❌ Local models slightly behind GPT-4/Claude-3.5 in reasoning
- ❌ Initial setup complexity
- ❌ Storage requirements (5-30GB per model)
- ❌ Limited to single-machine scaling
- ❌ User responsible for model updates

**Acceptable Compromises:**
- Quality gap narrowing rapidly (Qwen3, Gemma3, DeepSeek-V3)
- Hardware requirements reasonable for target audience (developers)
- Setup automated via scripts and clear documentation
- Storage cheap and getting cheaper
- Single-machine sufficient for experimentation and small-scale production

## Consequences

### Positive
- Users can experiment without cost concerns
- Privacy-first approach attracts security-conscious users
- Differentiates project from cloud-centric frameworks
- Enables use cases not possible with cloud APIs
- Aligns with growing local-AI movement
- Supports environmental sustainability (no data center traffic)

### Negative
- Smaller potential user base (requires capable hardware)
- Need to support multiple local model providers (Ollama, LM Studio)
- Documentation must cover hardware setup
- Performance varies by user hardware
- Cannot leverage latest GPT/Claude models as primary

### Mitigation Strategies
1. **Hardware Requirements**: Clearly document minimum specs, provide performance benchmarks
2. **Model Quality**: Curate best local models, test thoroughly, provide model selection guidance
3. **Setup Complexity**: Automated installation scripts, detailed troubleshooting guide
4. **Cloud Integration**: Optional cloud provider support for comparison/fallback
5. **Performance Variance**: Profile on multiple hardware configurations, optimize for Apple Silicon

## Alternatives Considered

### Alternative 1: Cloud-First Architecture
**Pros:**
- Access to most capable models (GPT-4, Claude-3.5)
- No hardware requirements
- Zero setup time
- Automatic updates

**Cons:**
- Ongoing costs ($20-200/month typical)
- Privacy concerns
- Network dependency
- Rate limits
- No transparency

**Why Rejected:** Contradicts core vision of privacy-preserving, cost-effective experimentation. Cloud APIs already well-served by existing frameworks.

### Alternative 2: Hybrid Cloud-Local
**Pros:**
- Best of both worlds
- Fallback options
- Performance flexibility

**Cons:**
- Complex configuration
- Inconsistent behavior
- Cost still a concern
- Privacy model unclear

**Why Rejected:** Complexity not justified. Better to be local-first with optional cloud rather than truly hybrid.

### Alternative 3: Server-Based Local
**Pros:**
- Centralized management
- Multi-user support
- Better resource utilization

**Cons:**
- Network latency reintroduced
- Server maintenance overhead
- Not truly "local"

**Why Rejected:** Defeats purpose of local-first. Users can deploy on server themselves if needed.

## Implementation

### Phase 1: Foundation (Current)
- Ollama integration with ChatOllama
- Basic examples with qwen3:8b, gemma3:12b
- Documentation on model selection
- Hardware requirements guide

### Phase 2: Optimization
- Apple Silicon Metal acceleration
- Model quantization support (Q4, Q5, Q8)
- Caching and batching optimizations
- Performance benchmarking tools

### Phase 3: Optional Cloud
- OpenAI provider as alternative
- Anthropic Claude integration
- Cloud/local comparison tools
- Cost tracking utilities

### Key Components
```python
# config/settings.py
class ModelProvider(Enum):
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"  # Optional
    ANTHROPIC = "anthropic"  # Optional

DEFAULT_PROVIDER = ModelProvider.OLLAMA  # Local-first default
ALLOW_CLOUD_FALLBACK = False  # Explicit opt-in required
```

## Verification

### Success Criteria
- [ ] All core examples run without internet connection
- [ ] No API keys required for basic functionality
- [ ] Setup completes in < 10 minutes on supported hardware
- [ ] Performance meets targets (20+ tok/s on M3 Max)
- [ ] Documentation clearly explains local-first approach

### Monitoring
- Track setup success rate via feedback
- Benchmark performance across hardware configurations
- Monitor model quality via user satisfaction
- Measure cost savings vs cloud alternatives

## References
- [Ollama Documentation](https://github.com/ollama/ollama)
- [LM Studio](https://lmstudio.ai/)
- [Local-First Software](https://www.inkandswitch.com/local-first/)
- [LangChain Local Models Guide](https://python.langchain.com/docs/guides/local_llms)
- [Apple Silicon ML Performance](https://github.com/ml-explore/mlx)
- Project Vision: `/plans/0-readme.md`
- Development Plan: `/plans/1-research-plan.md`

## Related ADRs
- ADR-006: Apple Silicon Optimization
- Future: ADR on cloud integration patterns

## Changelog
- 2025-10-26: Initial version - local-first architecture decision
