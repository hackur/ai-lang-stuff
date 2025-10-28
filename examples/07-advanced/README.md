# Advanced Multi-Modal AI Examples

This directory contains advanced examples demonstrating multi-modal AI capabilities using local models. All examples run completely on-device with no cloud dependencies.

## Overview

Multi-modal AI enables systems to understand and reason across different types of data:
- **Vision + Language**: Understanding images and answering visual questions
- **Audio + Language**: Transcribing and analyzing spoken content
- **Documents**: Comprehending PDFs with text, images, and tables
- **Cross-Modal RAG**: Unified search across text and images

## Examples

### 1. Vision Agent (`vision_agent.py`)

Intelligent agent with comprehensive vision capabilities using `qwen3-vl:8b`.

**Features:**
- Detailed image analysis and description
- Visual question answering
- Image comparison across multiple files
- Visual reasoning tasks
- Interactive Q&A sessions

**Prerequisites:**
```bash
ollama pull qwen3-vl:8b
ollama pull qwen3:8b
```

**Usage:**
```bash
# Analyze a single image
python vision_agent.py /path/to/image.jpg

# Ask specific question
python vision_agent.py /path/to/image.jpg --question "What colors are prominent?"

# Compare multiple images
python vision_agent.py /path/to/images_dir --compare

# Visual reasoning task
python vision_agent.py /path/to/image.jpg --reasoning "Count the number of people"

# Interactive session
python vision_agent.py /path/to/image.jpg --interactive
```

**Example Output:**
```
Initial Analysis:
--------------------------------------------------------------------------------
Main Subject: A modern office workspace with natural lighting
Setting/Context: Indoor office environment during daytime
Visual Elements:
  - Colors: Predominantly white and light gray with blue accents
  - Composition: Organized workspace with laptop and desk accessories
  - Lighting: Natural light from large window, creating soft shadows
  - Textures: Smooth desk surface, fabric chair, wooden floor
Details: Laptop showing code editor, coffee mug, potted plant, notebook
Interpretation: Professional, minimalist workspace designed for productivity
```

### 2. Audio Transcription Agent (`audio_transcription_agent.py`)

Agent architecture for audio processing and transcript analysis workflows.

**Features:**
- Transcript loading and parsing
- Comprehensive conversation analysis
- Speaker identification
- Topic extraction
- Sentiment analysis
- Key points extraction
- Question answering over transcripts

**Prerequisites:**
```bash
ollama pull qwen3:8b
```

**Usage:**
```bash
# Analyze transcript from file
python audio_transcription_agent.py --text-file transcript.txt

# Provide transcript directly
python audio_transcription_agent.py --text "Speaker 1: Hello..."

# Ask question about transcript
python audio_transcription_agent.py --text-file transcript.txt \
  --question "What was the main decision?"

# Generate concise summary
python audio_transcription_agent.py --text-file transcript.txt \
  --summary-style concise
```

**Example Output:**
```
COMPREHENSIVE AUDIO ANALYSIS
================================================================================

SUMMARY:
--------------------------------------------------------------------------------
This conversation covers a project status update meeting where the team
discusses progress on the new feature implementation. Key topics include
testing completion, deployment timeline, and resource allocation.

KEY POINTS:
--------------------------------------------------------------------------------
1. Initial feature implementation completed successfully
2. Testing phase is currently underway with positive results
3. Deployment scheduled for next week pending final approval
4. Additional resources needed for documentation
5. Follow-up meeting planned to review test results

SPEAKERS:
--------------------------------------------------------------------------------
  - Speaker 1 (Project Lead)
  - Speaker 2 (Development Team)

TOPICS DISCUSSED:
--------------------------------------------------------------------------------
  - Feature implementation progress
  - Testing and quality assurance
  - Deployment planning
  - Resource allocation

SENTIMENT ANALYSIS:
--------------------------------------------------------------------------------
Overall Sentiment: positive
Tone: professional, collaborative
Explanation: The conversation maintains a constructive tone with clear
communication between team members about project status.
```

### 3. Multi-Modal RAG (`multimodal_rag.py`)

Unified RAG system for cross-modal retrieval across text and images.

**Features:**
- Index text documents and images together
- Cross-modal retrieval (query text, get images and vice versa)
- Unified answer generation combining visual and textual sources
- Mixed-media knowledge bases
- Source attribution with media type

**Prerequisites:**
```bash
ollama pull qwen3-vl:8b
ollama pull qwen3:8b
ollama pull qwen3-embedding
```

**Usage:**
```bash
# Index directory with mixed content
python multimodal_rag.py /path/to/data_dir --index

# Query the knowledge base
python multimodal_rag.py /path/to/data_dir \
  --query "What does the architecture diagram show?"

# Interactive mode
python multimodal_rag.py /path/to/data_dir --interactive

# Rebuild index
python multimodal_rag.py /path/to/data_dir --rebuild --index
```

**Directory Structure:**
```
data_dir/
├── document.txt          # Text files
├── report.md             # Markdown files
├── diagram.png           # Images
├── chart.jpg             # Charts/graphs
└── screenshot.png        # Screenshots
```

**Example Output:**
```
Answer:
================================================================================
According to the architecture diagram (diagram.png), the system follows a
microservices architecture with three main components:

1. API Gateway (shown at the top) handles all incoming requests
2. Service Layer includes Authentication, Data Processing, and Storage services
3. Database Layer uses both SQL and NoSQL databases for different data types

The text documentation (document.txt) further explains that this design was
chosen for scalability and fault isolation.

Sources (5):
  1. [IMAGE] diagram.png
  2. [TEXT] document.txt
  3. [IMAGE] chart.jpg
  4. [TEXT] report.md
  5. [IMAGE] screenshot.png

[Confidence: high | Media types: image, text]
```

### 4. Document Understanding Agent (`document_understanding.py`)

Comprehensive document analysis for PDFs and text files.

**Features:**
- PDF text extraction
- Document structure analysis
- Section identification
- Title extraction
- Summary generation (multiple styles)
- Key points extraction
- Question answering over documents
- Metadata extraction

**Prerequisites:**
```bash
ollama pull qwen3:8b
pip install pypdf  # For PDF support
```

**Usage:**
```bash
# Comprehensive analysis
python document_understanding.py /path/to/document.pdf

# Generate summary only
python document_understanding.py /path/to/document.pdf --summary-only

# Executive summary
python document_understanding.py /path/to/document.pdf \
  --summary-only --summary-style executive

# Ask specific question
python document_understanding.py /path/to/document.pdf \
  --question "What are the main recommendations?"
```

**Example Output:**
```
DOCUMENT ANALYSIS
================================================================================

TITLE: Quarterly Performance Report - Q4 2024
Pages: 15 | Format: pdf | Characters: 45,234

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
This quarterly report presents comprehensive performance metrics for Q4 2024,
highlighting significant growth in user engagement and revenue. The report
analyzes key performance indicators, market trends, and strategic initiatives
implemented during the quarter. Notable achievements include 25% revenue growth
and successful launch of three new product features. The document concludes
with recommendations for Q1 2025 focusing on market expansion and product
optimization.

--------------------------------------------------------------------------------
KEY POINTS
--------------------------------------------------------------------------------
1. Revenue increased 25% year-over-year, exceeding targets
2. User engagement metrics show 40% improvement in retention
3. Three major product features successfully launched
4. Customer satisfaction scores reached all-time high of 4.6/5
5. Market share grew by 3.2 percentage points
6. Operational costs reduced by 12% through efficiency improvements
7. Strategic partnerships established with two major enterprise clients

--------------------------------------------------------------------------------
STRUCTURE ANALYSIS
--------------------------------------------------------------------------------
Document type: Business Report
Main sections:
  - Executive Summary
  - Performance Metrics
  - Market Analysis
  - Product Updates
  - Financial Review
  - Recommendations
Organization: Follows standard business report format with clear sections,
data visualizations, and supporting tables.

--------------------------------------------------------------------------------
SECTIONS (15)
--------------------------------------------------------------------------------
1. Executive Summary (Page 1)
2. Key Performance Indicators (Page 2)
3. Revenue Analysis (Page 3)
4. User Engagement Metrics (Page 4)
5. Product Feature Launches (Page 5)
... and 10 more sections
```

## Model Requirements

All examples use local models via Ollama:

| Model | Size | Purpose | Required For |
|-------|------|---------|--------------|
| qwen3-vl:8b | ~8GB | Vision + language understanding | Vision agent, Multi-modal RAG |
| qwen3:8b | ~8GB | Text generation and reasoning | All examples |
| qwen3-embedding | ~500MB | Text embeddings for search | Multi-modal RAG |

**Installation:**
```bash
# Vision model (required for visual tasks)
ollama pull qwen3-vl:8b

# Text model (required for all examples)
ollama pull qwen3:8b

# Embedding model (required for RAG)
ollama pull qwen3-embedding
```

## System Requirements

### Minimum
- **RAM**: 16GB (for 8B models)
- **Storage**: 20GB free space
- **GPU**: Recommended but not required (Apple Silicon or CUDA)

### Recommended
- **RAM**: 32GB (for running multiple models)
- **Storage**: 50GB free space
- **GPU**: Apple M1/M2/M3 or NVIDIA GPU with 8GB+ VRAM

## Performance Tips

1. **Model Selection**
   - Use smaller models (qwen3:4b) for faster processing
   - Use vision model only when needed
   - Cache model responses for repeated queries

2. **Batch Processing**
   - Process multiple images/documents in batches
   - Use indexing for large collections
   - Enable persistent vector stores

3. **Resource Management**
   - Close unused model sessions
   - Monitor memory usage
   - Use GPU acceleration when available

## Error Handling

All examples include comprehensive error handling:

**Common Issues:**

```bash
# Ollama not running
Error: Ollama not running! Start with: ollama serve

# Model not available
Error: Model 'qwen3-vl:8b' not available
Install with: ollama pull qwen3-vl:8b

# File not found
Error: Image not found: /path/to/image.jpg

# Invalid format
Error: Invalid image format: .bmp
```

## Integration Patterns

### Using Vision Agent in Your Code

```python
from examples.advanced.vision_agent import VisionAgent
from pathlib import Path

# Initialize agent
agent = VisionAgent(
    vision_model="qwen3-vl:8b",
    text_model="qwen3:8b"
)

# Analyze image
result = agent.analyze_image(
    image_path=Path("photo.jpg"),
    detailed=True
)
print(result.content)

# Answer question
answer = agent.answer_question(
    image_path=Path("diagram.png"),
    question="What does this diagram represent?"
)
print(answer.content)
```

### Using Multi-Modal RAG in Your Code

```python
from examples.advanced.multimodal_rag import MultiModalRAG
from pathlib import Path

# Initialize RAG system
rag = MultiModalRAG(
    vision_model="qwen3-vl:8b",
    text_model="qwen3:8b",
    embedding_model="qwen3-embedding"
)

# Index directory
vectorstore, docs = rag.index_directory(
    data_dir=Path("knowledge_base"),
    collection_name="my_kb",
    persist_dir="./data/vector_stores"
)

# Query
result = rag.answer_with_sources(
    vectorstore=vectorstore,
    query="What are the main components?",
    k=5
)

print(result.answer)
for source in result.sources:
    print(f"- [{source.metadata['type']}] {source.metadata['filename']}")
```

### Using Document Agent in Your Code

```python
from examples.advanced.document_understanding import DocumentUnderstandingAgent
from pathlib import Path

# Initialize agent
agent = DocumentUnderstandingAgent(
    text_model="qwen3:8b"
)

# Comprehensive analysis
analysis = agent.comprehensive_analysis(
    file_path=Path("report.pdf")
)

print(f"Title: {analysis.title}")
print(f"Summary: {analysis.summary}")
print(f"Key Points: {len(analysis.key_points)}")

# Answer question
if analysis.metadata['pages'] > 0:
    pages = agent.load_pdf_text(Path("report.pdf"))
    answer = agent.answer_question(
        pages=pages,
        question="What are the recommendations?"
    )
    print(answer)
```

## Best Practices

### 1. Image Quality
- Use high-resolution images for better analysis
- Ensure good lighting and clarity
- Crop to relevant content when possible

### 2. Document Preparation
- Use searchable PDFs (not scanned images)
- Organize documents with clear structure
- Include descriptive filenames

### 3. Performance Optimization
- Index large collections once, reuse vector stores
- Use appropriate k values for retrieval (3-5 typically)
- Enable GPU acceleration in Ollama

### 4. Error Recovery
- Implement retry logic for model timeouts
- Validate inputs before processing
- Handle partial failures gracefully

## Advanced Use Cases

### 1. Visual Documentation Search
Combine document understanding with vision RAG to search technical documentation with diagrams.

### 2. Multi-Language Support
Use vision models to process documents in multiple languages (qwen3-vl supports 140+ languages).

### 3. Automated Report Generation
Extract data from multiple sources and generate comprehensive reports.

### 4. Knowledge Base QA
Build interactive Q&A systems over mixed-media knowledge bases.

## Troubleshooting

### Vision Model Issues

```bash
# Model loading slowly
# Solution: Ensure sufficient RAM, close other applications

# Poor image analysis quality
# Solution: Use higher resolution images, try different prompts

# Vision model not responding
# Solution: Check Ollama logs: ollama logs
```

### RAG System Issues

```bash
# Poor retrieval results
# Solution: Increase k value, improve document descriptions

# Vector store errors
# Solution: Delete and rebuild: rm -rf ./data/vector_stores/*

# Out of memory during indexing
# Solution: Process in smaller batches, reduce chunk size
```

### Document Processing Issues

```bash
# PDF extraction failing
# Solution: Install pypdf: pip install pypdf

# Encoding errors
# Solution: Ensure UTF-8 encoding, check file format

# Slow processing
# Solution: Process pages in parallel, cache results
```

## Future Enhancements

Potential improvements for these examples:

1. **Streaming Support**: Add streaming for real-time responses
2. **Batch Processing**: Parallel processing of multiple files
3. **Advanced OCR**: Integration with Tesseract for scanned documents
4. **Video Analysis**: Frame extraction and temporal understanding
5. **Graph Integration**: Visual knowledge graphs from documents
6. **Export Options**: Save analyses in various formats (JSON, HTML, Markdown)

## Related Examples

- **04-rag/vision_rag.py**: Basic vision RAG implementation
- **06-production/**: Production-ready patterns for deployment
- **03-multi-agent/**: Agent orchestration patterns

## Contributing

When adding new multi-modal examples:

1. Follow the established code structure
2. Include comprehensive docstrings
3. Add error handling and logging
4. Provide usage examples
5. Update this README
6. Test with multiple model configurations

## License

Part of the ai-lang-stuff local-first AI toolkit. See main README for license information.

## Support

For issues or questions:
1. Check Ollama documentation: https://ollama.ai/docs
2. Review LangChain guides: https://python.langchain.com/
3. Consult model documentation for qwen3-vl capabilities

---

**Note**: All examples are designed for local execution. No data is sent to external services. Your documents, images, and analyses remain private on your device.
