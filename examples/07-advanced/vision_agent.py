"""
Vision Agent with Multi-Modal Capabilities.

This example demonstrates an intelligent agent that can:
1. Process and understand images using vision-language models
2. Answer questions about visual content
3. Generate detailed image descriptions
4. Perform visual reasoning and comparison
5. Combine text and image understanding

Prerequisites:
- Ollama server running: `ollama serve`
- Models downloaded:
  - `ollama pull qwen3-vl:8b`  (vision-language model)
  - `ollama pull qwen3:8b`  (text model for planning)

Expected output:
Comprehensive image analysis, visual Q&A, and multi-modal reasoning capabilities.

Usage:
    python vision_agent.py /path/to/image.jpg
    python vision_agent.py /path/to/image.jpg --question "What colors are in this image?"
    python vision_agent.py /path/to/images_dir --compare  # Compare multiple images
"""

import base64
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Add project root to path for utils imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import OllamaManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from vision model analysis."""

    content: str
    image_path: Optional[Path] = None
    metadata: Optional[Dict] = None


class VisionAgent:
    """
    Intelligent agent with vision capabilities.

    This agent can:
    - Analyze images with detailed understanding
    - Answer specific questions about images
    - Compare multiple images
    - Generate structured descriptions
    - Perform visual reasoning tasks
    """

    def __init__(
        self,
        vision_model: str = "qwen3-vl:8b",
        text_model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize vision agent.

        Args:
            vision_model: Vision-language model name.
            text_model: Text-only model for reasoning.
            base_url: Ollama API endpoint.
        """
        self.vision_model = vision_model
        self.text_model = text_model
        self.base_url = base_url

        # Initialize vision model
        self.vision_llm = ChatOllama(model=vision_model, base_url=base_url, temperature=0.3)

        # Initialize text model for planning and reasoning
        self.text_llm = ChatOllama(model=text_model, base_url=base_url, temperature=0.7)

        logger.info(f"Initialized VisionAgent with models: {vision_model}, {text_model}")

    def encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 encoded image string.

        Raises:
            FileNotFoundError: If image doesn't exist.
            ValueError: If file is not a valid image.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Validate image extension
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        if image_path.suffix.lower() not in valid_extensions:
            raise ValueError(f"Invalid image format: {image_path.suffix}")

        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                logger.debug(f"Encoded image: {image_path.name}")
                return encoded
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")

    def analyze_image(
        self, image_path: Path, prompt: Optional[str] = None, detailed: bool = True
    ) -> VisionResult:
        """
        Analyze an image with optional custom prompt.

        Args:
            image_path: Path to image file.
            prompt: Optional custom analysis prompt.
            detailed: Whether to generate detailed analysis.

        Returns:
            VisionResult with analysis.
        """
        if detailed:
            default_prompt = """Analyze this image comprehensively. Include:

1. **Main Subject**: What is the primary focus of the image?
2. **Setting/Context**: Where is this? What's the environment?
3. **Visual Elements**:
   - Colors and color scheme
   - Composition and layout
   - Lighting and mood
   - Textures and patterns
4. **Details**: Any text, numbers, logos, or significant details
5. **Interpretation**: What story or message does this convey?

Be specific, thorough, and objective."""
        else:
            default_prompt = "Describe this image concisely, focusing on the main elements."

        prompt_text = prompt or default_prompt

        try:
            # Encode image
            image_b64 = self.encode_image(image_path)

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ]
            )

            # Get analysis
            logger.info(f"Analyzing image: {image_path.name}")
            response = self.vision_llm.invoke([message])

            return VisionResult(
                content=response.content,
                image_path=image_path,
                metadata={
                    "model": self.vision_model,
                    "prompt_type": "detailed" if detailed else "concise",
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            raise

    def answer_question(
        self, image_path: Path, question: str, context: Optional[str] = None
    ) -> VisionResult:
        """
        Answer a specific question about an image.

        Args:
            image_path: Path to image file.
            question: Question to answer.
            context: Optional additional context.

        Returns:
            VisionResult with answer.
        """
        try:
            image_b64 = self.encode_image(image_path)

            # Build prompt with context if provided
            prompt_parts = []
            if context:
                prompt_parts.append(f"Context: {context}\n")
            prompt_parts.append(f"Question: {question}")
            prompt_parts.append("\nAnswer based on the image. Be specific and accurate.")

            full_prompt = "\n".join(prompt_parts)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ]
            )

            logger.info(f"Answering question about {image_path.name}")
            response = self.vision_llm.invoke([message])

            return VisionResult(
                content=response.content,
                image_path=image_path,
                metadata={
                    "model": self.vision_model,
                    "question": question,
                    "has_context": context is not None,
                },
            )

        except Exception as e:
            logger.error(f"Error answering question for {image_path}: {e}")
            raise

    def compare_images(
        self, image_paths: List[Path], comparison_aspects: Optional[List[str]] = None
    ) -> VisionResult:
        """
        Compare multiple images and generate analysis.

        Args:
            image_paths: List of image paths to compare.
            comparison_aspects: Specific aspects to compare (e.g., ['color', 'composition']).

        Returns:
            VisionResult with comparison analysis.
        """
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images to compare")

        if len(image_paths) > 4:
            logger.warning("Comparing more than 4 images may be slow")

        try:
            # First, analyze each image individually
            logger.info(f"Analyzing {len(image_paths)} images for comparison...")
            individual_analyses = []

            for i, img_path in enumerate(image_paths, 1):
                logger.info(f"  Analyzing image {i}/{len(image_paths)}: {img_path.name}")
                result = self.analyze_image(img_path, detailed=False)
                individual_analyses.append(
                    {"filename": img_path.name, "description": result.content}
                )

            # Build comparison prompt
            aspects_text = ""
            if comparison_aspects:
                aspects_text = f"\nFocus on these aspects: {', '.join(comparison_aspects)}"

            comparison_prompt = f"""Compare these {len(image_paths)} images:

"""
            for i, analysis in enumerate(individual_analyses, 1):
                comparison_prompt += (
                    f"Image {i} ({analysis['filename']}):\n{analysis['description']}\n\n"
                )

            comparison_prompt += f"""Now provide a comprehensive comparison:{aspects_text}

Include:
1. Similarities across images
2. Key differences
3. Unique features of each image
4. Overall assessment

Format as a structured comparison."""

            # Use text model for comparison reasoning
            logger.info("Generating comparison analysis...")
            response = self.text_llm.invoke([HumanMessage(content=comparison_prompt)])

            return VisionResult(
                content=response.content,
                metadata={
                    "model": self.text_model,
                    "images": [str(p) for p in image_paths],
                    "num_images": len(image_paths),
                    "aspects": comparison_aspects,
                },
            )

        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            raise

    def visual_reasoning(self, image_path: Path, reasoning_task: str) -> VisionResult:
        """
        Perform complex visual reasoning tasks.

        Args:
            image_path: Path to image file.
            reasoning_task: Description of reasoning task.

        Returns:
            VisionResult with reasoning analysis.

        Examples:
            - "Count the number of people in this image"
            - "Identify safety hazards visible in this image"
            - "Determine the time of day based on lighting"
        """
        try:
            image_b64 = self.encode_image(image_path)

            reasoning_prompt = f"""Task: {reasoning_task}

Approach this systematically:
1. Observe relevant visual elements
2. Apply logical reasoning
3. Provide step-by-step explanation
4. State your conclusion clearly

Be thorough and explain your reasoning process."""

            message = HumanMessage(
                content=[
                    {"type": "text", "text": reasoning_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ]
            )

            logger.info(f"Performing visual reasoning on {image_path.name}")
            response = self.vision_llm.invoke([message])

            return VisionResult(
                content=response.content,
                image_path=image_path,
                metadata={
                    "model": self.vision_model,
                    "task": reasoning_task,
                    "task_type": "visual_reasoning",
                },
            )

        except Exception as e:
            logger.error(f"Error in visual reasoning: {e}")
            raise

    def interactive_session(self, image_path: Path):
        """
        Run interactive Q&A session for a single image.

        Args:
            image_path: Path to image to analyze.
        """
        print("\n" + "=" * 80)
        print("Vision Agent - Interactive Session")
        print(f"Image: {image_path.name}")
        print("=" * 80)

        # Initial analysis
        print("\nGenerating initial analysis...\n")
        try:
            initial = self.analyze_image(image_path, detailed=True)
            print("Initial Analysis:")
            print("-" * 80)
            print(initial.content)
            print("-" * 80)
        except Exception as e:
            print(f"Error during initial analysis: {e}")
            return

        print("\n\nAsk questions about this image. Type 'quit' to exit.")
        print("Special commands:")
        print("  'analyze' - Generate new detailed analysis")
        print("  'describe' - Get concise description")
        print("  'quit' - Exit session")
        print("=" * 80 + "\n")

        conversation_history = []

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nEnding session. Goodbye!")
                    break

                if user_input.lower() == "analyze":
                    print("\nGenerating detailed analysis...")
                    result = self.analyze_image(image_path, detailed=True)
                    print("\nAnalysis:")
                    print("-" * 60)
                    print(result.content)
                    print("-" * 60)
                    continue

                if user_input.lower() == "describe":
                    print("\nGenerating description...")
                    result = self.analyze_image(image_path, detailed=False)
                    print("\nDescription:")
                    print("-" * 60)
                    print(result.content)
                    print("-" * 60)
                    continue

                # Build context from conversation history
                context = None
                if conversation_history:
                    context = "Previous conversation:\n" + "\n".join(
                        f"Q: {q}\nA: {a[:100]}..." for q, a in conversation_history[-3:]
                    )

                # Answer question
                print("\nThinking...")
                result = self.answer_question(
                    image_path=image_path, question=user_input, context=context
                )

                print("\nAgent:")
                print("-" * 60)
                print(result.content)
                print("-" * 60)

                # Store in history
                conversation_history.append((user_input, result.content))

            except KeyboardInterrupt:
                print("\n\nEnding session. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"\nError: {e}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision Agent for multi-modal image understanding")
    parser.add_argument("image_path", help="Path to image file or directory")
    parser.add_argument("--question", "-q", help="Ask a specific question about the image")
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple images (requires directory path)"
    )
    parser.add_argument("--reasoning", "-r", help="Perform visual reasoning task")
    parser.add_argument(
        "--vision-model", default="qwen3-vl:8b", help="Vision model (default: qwen3-vl:8b)"
    )
    parser.add_argument("--text-model", default="qwen3:8b", help="Text model (default: qwen3:8b)")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive Q&A session"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Vision Agent - Initialization")
    print("=" * 80)

    # Check Ollama
    print("\n1. Checking Ollama...")
    ollama_mgr = OllamaManager()

    if not ollama_mgr.check_ollama_running():
        print("Error: Ollama not running! Start with: ollama serve")
        return

    # Verify models
    print("\n2. Verifying models...")
    required_models = [args.vision_model, args.text_model]
    for model_name in required_models:
        print(f"   - {model_name}")
        if not ollama_mgr.ensure_model_available(model_name):
            print(f"Error: Model '{model_name}' not available")
            print(f"Install with: ollama pull {model_name}")
            return

    # Initialize agent
    print("\n3. Initializing vision agent...")
    agent = VisionAgent(vision_model=args.vision_model, text_model=args.text_model)

    # Determine mode
    input_path = Path(args.image_path)

    if not input_path.exists():
        print(f"\nError: Path not found: {args.image_path}")
        return

    try:
        # Compare mode
        if args.compare:
            if not input_path.is_dir():
                print("Error: --compare requires a directory path")
                return

            # Find all images in directory
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            images = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            if len(images) < 2:
                print(f"Error: Found {len(images)} images. Need at least 2 for comparison.")
                return

            print(f"\n4. Comparing {len(images)} images...")
            result = agent.compare_images(images[:4])  # Limit to 4 for performance

            print("\nComparison Analysis:")
            print("=" * 80)
            print(result.content)
            print("=" * 80)

        # Single image modes
        elif input_path.is_file():
            image_path = input_path

            # Interactive mode
            if args.interactive:
                agent.interactive_session(image_path)

            # Visual reasoning
            elif args.reasoning:
                print("\n4. Performing visual reasoning...")
                result = agent.visual_reasoning(image_path, args.reasoning)

                print("\nVisual Reasoning Analysis:")
                print("=" * 80)
                print(result.content)
                print("=" * 80)

            # Specific question
            elif args.question:
                print("\n4. Answering question...")
                result = agent.answer_question(image_path, args.question)

                print(f"\nQuestion: {args.question}")
                print("Answer:")
                print("=" * 80)
                print(result.content)
                print("=" * 80)

            # Default: detailed analysis
            else:
                print("\n4. Analyzing image...")
                result = agent.analyze_image(image_path, detailed=True)

                print("\nImage Analysis:")
                print("=" * 80)
                print(result.content)
                print("=" * 80)

        else:
            print("Error: Provide either an image file or directory with --compare")
            return

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\nError: {e}")
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
