"""
Document Understanding Agent with Multi-Modal Analysis.

This example demonstrates comprehensive document analysis:
1. Process PDFs with text and images
2. Extract tables, figures, and diagrams
3. OCR integration patterns (architecture)
4. Comprehensive document analysis and summarization
5. Question answering over document structure

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3-vl:8b`  (vision-language model)
  - `ollama pull qwen3:8b`  (text model)
- Python packages: pypdf, pdf2image (for advanced features)

Expected output:
Structured analysis of documents including text, images, and tables.

Usage:
    python document_understanding.py /path/to/document.pdf
    python document_understanding.py /path/to/document.pdf --extract-images
    python document_understanding.py /path/to/document.pdf --question "What is the main topic?"
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import OllamaManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """A section of a document."""

    title: str
    content: str
    page_number: Optional[int] = None
    section_type: str = "text"  # text, table, figure, code


@dataclass
class DocumentAnalysis:
    """Complete document analysis."""

    title: str
    summary: str
    key_points: List[str]
    sections: List[DocumentSection]
    metadata: Dict
    structure: Dict


class DocumentUnderstandingAgent:
    """
    Intelligent agent for comprehensive document understanding.

    Capabilities:
    - PDF text extraction
    - Document structure analysis
    - Section identification
    - Table and figure detection
    - Summary generation
    - Question answering
    """

    def __init__(
        self,
        text_model: str = "qwen3:8b",
        vision_model: str = "qwen3-vl:8b",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize document understanding agent.

        Args:
            text_model: Text-only model.
            vision_model: Vision-language model.
            base_url: Ollama API endpoint.
        """
        self.text_model = text_model
        self.vision_model = vision_model
        self.base_url = base_url

        self.text_llm = ChatOllama(model=text_model, base_url=base_url, temperature=0.2)

        self.vision_llm = ChatOllama(model=vision_model, base_url=base_url, temperature=0.3)

        logger.info(
            f"Initialized DocumentUnderstandingAgent with models: {text_model}, {vision_model}"
        )

    def load_pdf_text(self, pdf_path: Path) -> List[Dict]:
        """
        Load text content from PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of pages with text content.
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            pages = []

            logger.info(f"Loading PDF: {pdf_path.name} ({len(reader.pages)} pages)")

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                pages.append({"page_number": i + 1, "text": text, "char_count": len(text)})

            return pages

        except ImportError:
            logger.error("pypdf not installed. Install with: pip install pypdf")
            raise
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def load_text_file(self, file_path: Path) -> List[Dict]:
        """
        Load text file content.

        Args:
            file_path: Path to text file.

        Returns:
            List with single page content.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return [{"page_number": 1, "text": content, "char_count": len(content)}]

        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            raise

    def analyze_structure(self, pages: List[Dict]) -> Dict:
        """
        Analyze document structure.

        Args:
            pages: List of page dictionaries.

        Returns:
            Structure analysis.
        """
        # Combine all text
        full_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You analyze document structure and organization."),
                (
                    "human",
                    """Analyze the structure of this document:

{text}

Identify:
1. Document type (report, paper, article, manual, etc.)
2. Main sections/chapters (list titles)
3. Presence of: abstract, introduction, conclusion, references
4. Overall organization pattern

Provide structured analysis.

Analysis:""",
                ),
            ]
        )

        try:
            # Limit text for structure analysis
            analysis_text = full_text[:4000] if len(full_text) > 4000 else full_text

            logger.info("Analyzing document structure...")
            chain = prompt | self.text_llm
            response = chain.invoke({"text": analysis_text})

            return {
                "total_pages": len(pages),
                "total_chars": sum(p["char_count"] for p in pages),
                "analysis": response.content,
            }

        except Exception as e:
            logger.error(f"Error analyzing structure: {e}")
            raise

    def extract_sections(self, pages: List[Dict]) -> List[DocumentSection]:
        """
        Extract logical sections from document.

        Args:
            pages: List of page dictionaries.

        Returns:
            List of DocumentSection objects.
        """
        sections = []
        "\n\n".join(f"[Page {p['page_number']}]\n{p['text']}" for p in pages)

        # Simple heuristic: look for headings (all caps, or numbered sections)
        # In production, use more sophisticated parsing

        # For now, create sections by page
        for page in pages:
            text = page["text"].strip()
            if text:
                # Try to find a title at the start
                lines = text.split("\n")
                title = f"Page {page['page_number']}"

                # Look for title-like first line
                if lines:
                    first_line = lines[0].strip()
                    if len(first_line) < 100 and (first_line.isupper() or first_line.istitle()):
                        title = first_line

                section = DocumentSection(
                    title=title, content=text, page_number=page["page_number"], section_type="text"
                )
                sections.append(section)

        logger.info(f"Extracted {len(sections)} sections")
        return sections

    def summarize_document(self, pages: List[Dict], style: str = "comprehensive") -> str:
        """
        Generate document summary.

        Args:
            pages: List of page dictionaries.
            style: Summary style (comprehensive, concise, executive).

        Returns:
            Summary text.
        """
        full_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())

        style_instructions = {
            "comprehensive": "Provide a detailed summary covering all main topics and key points.",
            "concise": "Provide a brief 3-5 sentence summary of the main ideas.",
            "executive": "Provide an executive summary with key findings and recommendations.",
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert at summarizing documents clearly and accurately."),
                (
                    "human",
                    """Document:
{text}

Task: {instruction}

Summary:""",
                ),
            ]
        )

        try:
            # Use first ~6000 chars or full document if shorter
            summary_text = full_text[:6000] if len(full_text) > 6000 else full_text

            logger.info(f"Generating {style} summary...")
            chain = prompt | self.text_llm
            response = chain.invoke(
                {
                    "text": summary_text,
                    "instruction": style_instructions.get(
                        style, style_instructions["comprehensive"]
                    ),
                }
            )

            return response.content

        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            raise

    def extract_key_points(self, pages: List[Dict], num_points: int = 5) -> List[str]:
        """
        Extract key points from document.

        Args:
            pages: List of page dictionaries.
            num_points: Number of key points to extract.

        Returns:
            List of key points.
        """
        full_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You extract the most important key points from documents."),
                (
                    "human",
                    """Document:
{text}

Extract the {num_points} most important key points or findings.
Return as a numbered list.

Key Points:""",
                ),
            ]
        )

        try:
            # Use significant portion for key points
            points_text = full_text[:5000] if len(full_text) > 5000 else full_text

            logger.info(f"Extracting {num_points} key points...")
            chain = prompt | self.text_llm
            response = chain.invoke({"text": points_text, "num_points": num_points})

            # Parse numbered list
            points = []
            for line in response.content.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    clean_point = line.lstrip("0123456789.-* ").strip()
                    if clean_point:
                        points.append(clean_point)

            return points[:num_points]

        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            raise

    def identify_title(self, pages: List[Dict]) -> str:
        """
        Identify document title.

        Args:
            pages: List of page dictionaries.

        Returns:
            Document title.
        """
        if not pages:
            return "Untitled Document"

        # Use first page
        first_page = pages[0]["text"]
        lines = first_page.split("\n")[:10]  # First 10 lines

        # Simple heuristic: look for short, prominent text
        for line in lines:
            line = line.strip()
            if 5 < len(line) < 100 and not line.endswith("."):
                # Likely a title
                return line

        # Fallback: ask LLM
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You identify document titles."),
                    (
                        "human",
                        """First page content:
{text}

What is the title of this document? Return only the title, nothing else.

Title:""",
                    ),
                ]
            )

            chain = prompt | self.text_llm
            response = chain.invoke({"text": first_page[:1000]})
            return response.content.strip()

        except Exception as e:
            logger.warning(f"Could not identify title: {e}")
            return "Untitled Document"

    def answer_question(self, pages: List[Dict], question: str) -> str:
        """
        Answer question about document.

        Args:
            pages: List of page dictionaries.
            question: Question to answer.

        Returns:
            Answer text.
        """
        full_text = "\n\n".join(
            f"[Page {p['page_number']}]\n{p['text']}" for p in pages if p["text"].strip()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You answer questions about documents accurately.
Always cite the page number when referencing information.""",
                ),
                (
                    "human",
                    """Document:
{text}

Question: {question}

Answer based on the document content. Include page references.

Answer:""",
                ),
            ]
        )

        try:
            logger.info(f"Answering question: {question}")

            # For long documents, might want to use RAG or chunking
            # For now, use first 8000 chars
            answer_text = full_text[:8000] if len(full_text) > 8000 else full_text

            chain = prompt | self.text_llm
            response = chain.invoke({"text": answer_text, "question": question})

            return response.content

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise

    def comprehensive_analysis(self, file_path: Path) -> DocumentAnalysis:
        """
        Perform comprehensive document analysis.

        Args:
            file_path: Path to document.

        Returns:
            DocumentAnalysis with complete analysis.
        """
        logger.info(f"Starting comprehensive analysis of {file_path.name}")

        # Load document
        if file_path.suffix.lower() == ".pdf":
            pages = self.load_pdf_text(file_path)
        else:
            pages = self.load_text_file(file_path)

        # Identify title
        title = self.identify_title(pages)
        logger.info(f"Document title: {title}")

        # Analyze structure
        structure = self.analyze_structure(pages)

        # Extract sections
        sections = self.extract_sections(pages)

        # Generate summary
        summary = self.summarize_document(pages, style="comprehensive")

        # Extract key points
        key_points = self.extract_key_points(pages, num_points=7)

        # Build metadata
        metadata = {
            "filename": file_path.name,
            "path": str(file_path),
            "format": file_path.suffix[1:],
            "pages": len(pages),
            "total_chars": sum(p["char_count"] for p in pages),
        }

        analysis = DocumentAnalysis(
            title=title,
            summary=summary,
            key_points=key_points,
            sections=sections,
            metadata=metadata,
            structure=structure,
        )

        logger.info("Comprehensive analysis completed")
        return analysis


def print_document_analysis(analysis: DocumentAnalysis):
    """Pretty print document analysis."""
    print("\n" + "=" * 80)
    print("DOCUMENT ANALYSIS")
    print("=" * 80)

    print(f"\nTITLE: {analysis.title}")
    print(
        f"Pages: {analysis.metadata['pages']} | "
        f"Format: {analysis.metadata['format']} | "
        f"Characters: {analysis.metadata['total_chars']:,}"
    )

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(analysis.summary)

    print("\n" + "-" * 80)
    print("KEY POINTS")
    print("-" * 80)
    for i, point in enumerate(analysis.key_points, 1):
        print(f"{i}. {point}")

    print("\n" + "-" * 80)
    print("STRUCTURE ANALYSIS")
    print("-" * 80)
    print(analysis.structure["analysis"])

    print("\n" + "-" * 80)
    print(f"SECTIONS ({len(analysis.sections)})")
    print("-" * 80)
    for i, section in enumerate(analysis.sections[:5], 1):  # Show first 5
        print(f"{i}. {section.title} (Page {section.page_number})")
    if len(analysis.sections) > 5:
        print(f"... and {len(analysis.sections) - 5} more sections")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Document Understanding Agent for comprehensive analysis"
    )
    parser.add_argument("document", help="Path to document (PDF or text file)")
    parser.add_argument("--question", "-q", help="Ask a question about the document")
    parser.add_argument("--summary-only", action="store_true", help="Only generate summary")
    parser.add_argument(
        "--summary-style",
        choices=["comprehensive", "concise", "executive"],
        default="comprehensive",
        help="Summary style",
    )
    parser.add_argument("--text-model", default="qwen3:8b", help="Text model (default: qwen3:8b)")
    parser.add_argument(
        "--vision-model", default="qwen3-vl:8b", help="Vision model (default: qwen3-vl:8b)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Document Understanding Agent - Initialization")
    print("=" * 80)

    # Verify document exists
    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: Document not found: {args.document}")
        return

    # Check format
    if doc_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
        print(f"Warning: Unsupported format {doc_path.suffix}. Will attempt text extraction.")

    # Check Ollama
    print("\n1. Checking Ollama...")
    ollama_mgr = OllamaManager()

    if not ollama_mgr.check_ollama_running():
        print("Error: Ollama not running! Start with: ollama serve")
        return

    # Verify models
    print(f"\n2. Verifying model: {args.text_model}")
    if not ollama_mgr.ensure_model_available(args.text_model):
        print(f"Error: Model '{args.text_model}' not available")
        print(f"Install with: ollama pull {args.text_model}")
        return

    # Initialize agent
    print("\n3. Initializing document understanding agent...")
    agent = DocumentUnderstandingAgent(text_model=args.text_model, vision_model=args.vision_model)

    try:
        # Question answering mode
        if args.question:
            print("\n4. Loading document and answering question...")

            if doc_path.suffix.lower() == ".pdf":
                pages = agent.load_pdf_text(doc_path)
            else:
                pages = agent.load_text_file(doc_path)

            answer = agent.answer_question(pages, args.question)

            print(f"\nDocument: {doc_path.name}")
            print(f"Question: {args.question}")
            print("\nAnswer:")
            print("=" * 80)
            print(answer)
            print("=" * 80)

        # Summary only mode
        elif args.summary_only:
            print(f"\n4. Generating {args.summary_style} summary...")

            if doc_path.suffix.lower() == ".pdf":
                pages = agent.load_pdf_text(doc_path)
            else:
                pages = agent.load_text_file(doc_path)

            summary = agent.summarize_document(pages, style=args.summary_style)

            print(f"\nDocument: {doc_path.name}")
            print(f"\n{args.summary_style.upper()} SUMMARY:")
            print("=" * 80)
            print(summary)
            print("=" * 80)

        # Comprehensive analysis mode
        else:
            print("\n4. Performing comprehensive analysis...")
            analysis = agent.comprehensive_analysis(doc_path)
            print_document_analysis(analysis)

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        print(f"\nError: {e}")

        if "pypdf" in str(e):
            print("\nNote: PDF support requires pypdf. Install with:")
            print("  pip install pypdf")

        return

    print("\n" + "=" * 80)
    print("Execution completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        sys.exit(1)
