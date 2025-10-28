"""
Audio Transcription Agent with Multi-Modal Workflows.

This example demonstrates audio processing capabilities:
1. Audio transcription (using external tools or APIs)
2. Text analysis of transcribed content
3. Audio + text reasoning workflows
4. Conversation analysis and summarization

Note: This example focuses on the agent architecture for audio workflows.
For actual audio transcription, you would integrate tools like:
- Whisper (OpenAI's speech recognition)
- macOS built-in speech recognition
- External transcription services

Prerequisites:
- Ollama server running: `ollama serve`
- Model downloaded: `ollama pull qwen3:8b`
- (Optional) Whisper or other transcription tool

Expected output:
Architecture for audio transcription workflows with text analysis.

Usage:
    python audio_transcription_agent.py /path/to/audio.mp3
    python audio_transcription_agent.py --text-file transcript.txt  # Process existing transcript
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import json
from datetime import datetime

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
class TranscriptSegment:
    """A segment of transcribed audio."""

    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class AudioAnalysisResult:
    """Result from audio analysis."""

    summary: str
    key_points: List[str]
    speakers: List[str]
    topics: List[str]
    sentiment: str
    metadata: Dict


class AudioTranscriptionAgent:
    """
    Agent for audio transcription and analysis workflows.

    This agent can:
    - Process transcribed audio text
    - Analyze conversation content
    - Extract key insights and summaries
    - Identify speakers and topics
    - Perform sentiment analysis
    """

    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize audio transcription agent.

        Args:
            model: LLM model name.
            base_url: Ollama API endpoint.
        """
        self.model = model
        self.base_url = base_url

        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.3)

        logger.info(f"Initialized AudioTranscriptionAgent with model: {model}")

    def load_transcript(self, file_path: Path) -> str:
        """
        Load transcript from file.

        Args:
            file_path: Path to transcript file.

        Returns:
            Transcript text.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Loaded transcript from {file_path.name}")
            return content
        except Exception as e:
            logger.error(f"Error loading transcript: {e}")
            raise

    def parse_transcript_with_timestamps(self, content: str) -> List[TranscriptSegment]:
        """
        Parse transcript with timestamps into segments.

        Expected format:
        [00:00:12] Speaker 1: Hello there
        [00:00:15] Speaker 2: Hi, how are you?

        Args:
            content: Transcript content with timestamps.

        Returns:
            List of TranscriptSegment objects.
        """
        segments = []
        lines = content.strip().split("\n")

        for line in lines:
            # Simple parsing - extend for more complex formats
            if line.strip():
                # Try to extract timestamp and speaker
                # This is a simple implementation - customize as needed
                segments.append(TranscriptSegment(text=line))

        logger.info(f"Parsed {len(segments)} transcript segments")
        return segments

    def summarize_transcript(self, transcript: str, style: str = "concise") -> str:
        """
        Generate summary of transcript.

        Args:
            transcript: Full transcript text.
            style: Summary style ('concise', 'detailed', 'bullet_points').

        Returns:
            Summary text.
        """
        style_prompts = {
            "concise": "Provide a concise 2-3 sentence summary of the main points.",
            "detailed": "Provide a detailed summary covering all major topics discussed.",
            "bullet_points": "Summarize as a list of bullet points covering key topics.",
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at analyzing and summarizing conversations and audio transcripts.",
                ),
                (
                    "human",
                    """Transcript:
{transcript}

Task: {style_instruction}

Summary:""",
                ),
            ]
        )

        try:
            logger.info(f"Generating {style} summary...")
            chain = prompt | self.llm
            response = chain.invoke(
                {
                    "transcript": transcript,
                    "style_instruction": style_prompts.get(style, style_prompts["concise"]),
                }
            )

            return response.content

        except Exception as e:
            logger.error(f"Error summarizing transcript: {e}")
            raise

    def extract_key_points(self, transcript: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from transcript.

        Args:
            transcript: Full transcript text.
            num_points: Number of key points to extract.

        Returns:
            List of key points.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You extract the most important points from conversations."),
                (
                    "human",
                    """Transcript:
{transcript}

Extract the {num_points} most important key points. Return as a numbered list.

Key Points:""",
                ),
            ]
        )

        try:
            logger.info(f"Extracting {num_points} key points...")
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript, "num_points": num_points})

            # Parse numbered list
            content = response.content
            points = []
            for line in content.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    # Remove numbering/bullets
                    clean_point = line.lstrip("0123456789.-* ")
                    if clean_point:
                        points.append(clean_point)

            return points[:num_points]

        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            raise

    def identify_speakers(self, transcript: str) -> List[str]:
        """
        Identify speakers in transcript.

        Args:
            transcript: Full transcript text.

        Returns:
            List of identified speakers.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You identify and list speakers in conversation transcripts."),
                (
                    "human",
                    """Transcript:
{transcript}

Identify all speakers/participants in this conversation. If names are not mentioned, use descriptions like "Speaker 1", "Speaker 2", etc.

Return only the list of speakers, one per line.

Speakers:""",
                ),
            ]
        )

        try:
            logger.info("Identifying speakers...")
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript})

            speakers = [line.strip() for line in response.content.split("\n") if line.strip()]

            return speakers

        except Exception as e:
            logger.error(f"Error identifying speakers: {e}")
            raise

    def analyze_sentiment(self, transcript: str) -> Dict[str, str]:
        """
        Analyze sentiment of transcript.

        Args:
            transcript: Full transcript text.

        Returns:
            Dictionary with sentiment analysis.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You analyze the sentiment and emotional tone of conversations."),
                (
                    "human",
                    """Transcript:
{transcript}

Analyze the overall sentiment and emotional tone. Provide:
1. Overall sentiment (positive/neutral/negative)
2. Emotional tone (e.g., professional, casual, tense, friendly)
3. Brief explanation

Format as JSON:
{{
  "sentiment": "positive/neutral/negative",
  "tone": "description",
  "explanation": "brief explanation"
}}

Analysis:""",
                ),
            ]
        )

        try:
            logger.info("Analyzing sentiment...")
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript})

            # Try to parse JSON, fallback to text
            try:
                sentiment_data = json.loads(response.content)
            except json.JSONDecodeError:
                sentiment_data = {
                    "sentiment": "unknown",
                    "tone": "unknown",
                    "explanation": response.content,
                }

            return sentiment_data

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise

    def identify_topics(self, transcript: str, max_topics: int = 5) -> List[str]:
        """
        Identify main topics discussed in transcript.

        Args:
            transcript: Full transcript text.
            max_topics: Maximum number of topics to identify.

        Returns:
            List of topics.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You identify the main topics and themes in conversations."),
                (
                    "human",
                    """Transcript:
{transcript}

Identify the main topics or themes discussed. List up to {max_topics} topics.

Return only the topics, one per line.

Topics:""",
                ),
            ]
        )

        try:
            logger.info(f"Identifying topics (max {max_topics})...")
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript, "max_topics": max_topics})

            topics = [line.strip() for line in response.content.split("\n") if line.strip()]

            return topics[:max_topics]

        except Exception as e:
            logger.error(f"Error identifying topics: {e}")
            raise

    def comprehensive_analysis(
        self, transcript: str, include_sentiment: bool = True
    ) -> AudioAnalysisResult:
        """
        Perform comprehensive analysis of transcript.

        Args:
            transcript: Full transcript text.
            include_sentiment: Whether to include sentiment analysis.

        Returns:
            AudioAnalysisResult with complete analysis.
        """
        logger.info("Starting comprehensive analysis...")

        try:
            # Generate summary
            summary = self.summarize_transcript(transcript, style="detailed")

            # Extract key points
            key_points = self.extract_key_points(transcript, num_points=5)

            # Identify speakers
            speakers = self.identify_speakers(transcript)

            # Identify topics
            topics = self.identify_topics(transcript, max_topics=5)

            # Analyze sentiment (optional)
            sentiment_data = {}
            if include_sentiment:
                sentiment_data = self.analyze_sentiment(transcript)

            result = AudioAnalysisResult(
                summary=summary,
                key_points=key_points,
                speakers=speakers,
                topics=topics,
                sentiment=sentiment_data.get("sentiment", "not analyzed"),
                metadata={
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "transcript_length": len(transcript),
                    "sentiment_details": sentiment_data,
                },
            )

            logger.info("Comprehensive analysis completed")
            return result

        except Exception as e:
            logger.error(f"Error during comprehensive analysis: {e}")
            raise

    def answer_question(self, transcript: str, question: str) -> str:
        """
        Answer question about transcript content.

        Args:
            transcript: Full transcript text.
            question: Question to answer.

        Returns:
            Answer text.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer questions about conversation transcripts accurately and concisely.",
                ),
                (
                    "human",
                    """Transcript:
{transcript}

Question: {question}

Answer based on the transcript content. If the information isn't in the transcript, say so.

Answer:""",
                ),
            ]
        )

        try:
            logger.info(f"Answering question: {question}")
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript, "question": question})

            return response.content

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise


def print_analysis_results(result: AudioAnalysisResult):
    """Pretty print analysis results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE AUDIO ANALYSIS")
    print("=" * 80)

    print("\nSUMMARY:")
    print("-" * 80)
    print(result.summary)

    print("\n\nKEY POINTS:")
    print("-" * 80)
    for i, point in enumerate(result.key_points, 1):
        print(f"{i}. {point}")

    print("\n\nSPEAKERS:")
    print("-" * 80)
    for speaker in result.speakers:
        print(f"  - {speaker}")

    print("\n\nTOPICS DISCUSSED:")
    print("-" * 80)
    for topic in result.topics:
        print(f"  - {topic}")

    if result.metadata.get("sentiment_details"):
        sentiment = result.metadata["sentiment_details"]
        print("\n\nSENTIMENT ANALYSIS:")
        print("-" * 80)
        print(f"Overall Sentiment: {sentiment.get('sentiment', 'N/A')}")
        print(f"Tone: {sentiment.get('tone', 'N/A')}")
        print(f"Explanation: {sentiment.get('explanation', 'N/A')}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio Transcription Agent for transcript analysis"
    )
    parser.add_argument("--text-file", help="Path to transcript text file")
    parser.add_argument("--text", help="Transcript text directly")
    parser.add_argument("--question", "-q", help="Ask a question about the transcript")
    parser.add_argument(
        "--summary-style",
        choices=["concise", "detailed", "bullet_points"],
        default="detailed",
        help="Summary style (default: detailed)",
    )
    parser.add_argument("--model", default="qwen3:8b", help="LLM model (default: qwen3:8b)")
    parser.add_argument("--no-sentiment", action="store_true", help="Skip sentiment analysis")

    args = parser.parse_args()

    print("=" * 80)
    print("Audio Transcription Agent - Initialization")
    print("=" * 80)

    # Check Ollama
    print("\n1. Checking Ollama...")
    ollama_mgr = OllamaManager()

    if not ollama_mgr.check_ollama_running():
        print("Error: Ollama not running! Start with: ollama serve")
        return

    # Verify model
    print(f"\n2. Verifying model: {args.model}")
    if not ollama_mgr.ensure_model_available(args.model):
        print(f"Error: Model '{args.model}' not available")
        print(f"Install with: ollama pull {args.model}")
        return

    # Initialize agent
    print("\n3. Initializing audio transcription agent...")
    agent = AudioTranscriptionAgent(model=args.model)

    # Load transcript
    transcript = None
    if args.text_file:
        print(f"\n4. Loading transcript from file: {args.text_file}")
        try:
            transcript = agent.load_transcript(Path(args.text_file))
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    elif args.text:
        print("\n4. Using provided transcript text")
        transcript = args.text
    else:
        print("\nError: Provide either --text-file or --text")
        print("\nExample transcript format:")
        print("-" * 60)
        example = """Speaker 1: Welcome to today's meeting about project updates.
Speaker 2: Thanks for having me. I'd like to start by discussing our progress on the new feature.
Speaker 1: Great, please go ahead.
Speaker 2: We've completed the initial implementation and testing is underway."""
        print(example)
        print("-" * 60)
        return

    if not transcript or not transcript.strip():
        print("Error: Empty transcript")
        return

    try:
        # Process based on mode
        if args.question:
            # Question answering mode
            print("\n5. Answering question...")
            answer = agent.answer_question(transcript, args.question)

            print(f"\nQuestion: {args.question}")
            print("\nAnswer:")
            print("-" * 80)
            print(answer)
            print("-" * 80)

        else:
            # Comprehensive analysis mode
            print("\n5. Performing comprehensive analysis...")
            result = agent.comprehensive_analysis(
                transcript, include_sentiment=not args.no_sentiment
            )

            print_analysis_results(result)

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
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
