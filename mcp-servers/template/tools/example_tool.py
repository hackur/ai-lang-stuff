"""
Example Tool Implementation

This file demonstrates the standard pattern for implementing MCP server tools.
Copy and modify this template for your own tools.

Tool Structure:
1. Input validation using Pydantic models
2. Business logic implementation
3. Output formatting
4. Error handling
5. Logging

Best Practices:
- Use type hints for all parameters
- Validate inputs with Pydantic
- Return structured data (dict or Pydantic model)
- Handle errors gracefully
- Log important operations
- Include docstrings with examples
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================================
# Input/Output Models
# ============================================================================


class GreetingRequest(BaseModel):
    """Model for greeting request parameters."""

    name: str = Field(..., description="Name of person to greet", min_length=1)
    greeting: str = Field(
        default="Hello",
        description="Greeting to use",
    )
    formal: bool = Field(
        default=False,
        description="Whether to use formal greeting",
    )

    @validator("name")
    def validate_name(cls, v):
        """Validate name contains only letters and spaces."""
        if not v.replace(" ", "").isalpha():
            raise ValueError("Name must contain only letters and spaces")
        return v.strip()

    @validator("greeting")
    def validate_greeting(cls, v):
        """Validate greeting is not empty."""
        if not v.strip():
            raise ValueError("Greeting cannot be empty")
        return v.strip()


class GreetingResponse(BaseModel):
    """Model for greeting response."""

    message: str = Field(..., description="Greeting message")
    name: str = Field(..., description="Name that was greeted")
    formal: bool = Field(..., description="Whether formal greeting was used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# ============================================================================
# Tool Implementation
# ============================================================================


def greet(name: str, greeting: str = "Hello", formal: bool = False) -> Dict[str, Any]:
    """
    Generate a greeting message.

    This is an example tool that demonstrates the standard pattern:
    1. Validate inputs using Pydantic
    2. Execute business logic
    3. Return structured output
    4. Handle errors appropriately

    Args:
        name: Name of person to greet (required)
        greeting: Greeting to use (default: "Hello")
        formal: Whether to use formal greeting (default: False)

    Returns:
        Dict with greeting message and metadata

    Raises:
        ValueError: If validation fails

    Example:
        >>> result = greet("Alice", formal=True)
        >>> print(result["message"])
        Good day, Ms. Alice.

        >>> result = greet("Bob", greeting="Hi")
        >>> print(result["message"])
        Hi, Bob!
    """
    try:
        # 1. Validate inputs
        request = GreetingRequest(
            name=name,
            greeting=greeting,
            formal=formal,
        )

        logger.info(f"Generating greeting for: {request.name}")

        # 2. Business logic
        if request.formal:
            # Formal greeting
            honorific = "Mr./Ms."
            punctuation = "."
            message = f"Good day, {honorific} {request.name}{punctuation}"
        else:
            # Informal greeting
            message = f"{request.greeting}, {request.name}!"

        # 3. Format output
        response = GreetingResponse(
            message=message,
            name=request.name,
            formal=request.formal,
            metadata={
                "greeting_used": request.greeting,
                "message_length": len(message),
            },
        )

        logger.info(f"Generated greeting: {message}")

        # Return as dict for MCP server
        return response.model_dump()

    except Exception as e:
        # 4. Error handling
        logger.error(f"Error generating greeting: {e}")
        raise


# ============================================================================
# More Complex Example: Text Analysis Tool
# ============================================================================


class TextAnalysisRequest(BaseModel):
    """Model for text analysis request."""

    text: str = Field(..., description="Text to analyze", min_length=1)
    include_words: bool = Field(
        default=True,
        description="Include word analysis",
    )
    include_sentences: bool = Field(
        default=True,
        description="Include sentence analysis",
    )


class TextAnalysisResponse(BaseModel):
    """Model for text analysis response."""

    character_count: int = Field(..., description="Total characters")
    word_count: int = Field(..., description="Total words")
    sentence_count: int = Field(..., description="Total sentences")
    average_word_length: float = Field(..., description="Average word length")
    most_common_words: Optional[List[tuple[str, int]]] = Field(
        default=None,
        description="Most common words and their counts",
    )
    longest_sentence: Optional[str] = Field(
        default=None,
        description="Longest sentence",
    )


def analyze_text(
    text: str,
    include_words: bool = True,
    include_sentences: bool = True,
) -> Dict[str, Any]:
    """
    Analyze text and return statistics.

    Demonstrates a more complex tool with:
    - Multiple analysis steps
    - Optional features
    - Complex output structure

    Args:
        text: Text to analyze
        include_words: Whether to include word analysis
        include_sentences: Whether to include sentence analysis

    Returns:
        Dict with text statistics

    Example:
        >>> text = "Hello world. This is a test. Testing analysis."
        >>> result = analyze_text(text)
        >>> print(f"Words: {result['word_count']}")
        Words: 8
    """
    try:
        # Validate input
        request = TextAnalysisRequest(
            text=text,
            include_words=include_words,
            include_sentences=include_sentences,
        )

        logger.info(f"Analyzing text ({len(request.text)} chars)")

        # Basic analysis
        character_count = len(request.text)
        words = request.text.split()
        word_count = len(words)

        # Sentence analysis
        sentences = [s.strip() for s in request.text.split(".") if s.strip()]
        sentence_count = len(sentences)

        # Word analysis
        average_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

        most_common_words = None
        if include_words and words:
            # Count word frequency
            word_freq: Dict[str, int] = {}
            for word in words:
                word_lower = word.lower().strip(".,!?;:")
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1

            # Get top 5 most common
            most_common_words = sorted(
                word_freq.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

        # Sentence analysis
        longest_sentence = None
        if include_sentences and sentences:
            longest_sentence = max(sentences, key=len)

        # Build response
        response = TextAnalysisResponse(
            character_count=character_count,
            word_count=word_count,
            sentence_count=sentence_count,
            average_word_length=round(average_word_length, 2),
            most_common_words=most_common_words,
            longest_sentence=longest_sentence,
        )

        logger.info(f"Analysis complete: {word_count} words, {sentence_count} sentences")

        return response.model_dump()

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise


# ============================================================================
# Tool with External Dependencies
# ============================================================================


class DataFetchRequest(BaseModel):
    """Model for data fetch request."""

    url: str = Field(..., description="URL to fetch data from")
    timeout: int = Field(
        default=30,
        description="Timeout in seconds",
        ge=1,
        le=300,
    )

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


def fetch_data(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch data from URL.

    Demonstrates tool with external dependencies:
    - HTTP requests
    - Timeouts
    - Error handling for network issues

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Dict with fetched data and metadata

    Example:
        >>> result = fetch_data("https://api.example.com/data")
        >>> print(result["status_code"])
        200

    Note:
        This is a template - actual implementation would use httpx or requests.
    """
    try:
        # Validate input
        request = DataFetchRequest(url=url, timeout=timeout)

        logger.info(f"Fetching data from: {request.url}")

        # Simulated fetch (in real implementation, use httpx)
        # import httpx
        # response = httpx.get(request.url, timeout=request.timeout)

        # Simulated response
        response_data = {
            "status_code": 200,
            "url": request.url,
            "data": "Simulated response data",
            "headers": {"content-type": "application/json"},
            "elapsed_ms": 150,
        }

        logger.info(f"Fetched data successfully: {response_data['status_code']}")

        return response_data

    except Exception as e:
        logger.error(f"Error fetching data from {url}: {e}")
        raise


# ============================================================================
# Tool Registration Helper
# ============================================================================


def register_example_tools(server):
    """
    Register all example tools with an MCP server.

    This is a helper function to make tool registration cleaner.

    Args:
        server: MCPServer instance

    Example:
        >>> from server import MCPServer
        >>> server = MCPServer("my-server")
        >>> register_example_tools(server)
        >>> print(len(server.list_tools()))
        3
    """
    from server import ToolParameter

    # Register greeting tool
    server.register_tool(
        name="greet",
        function=greet,
        description="Generate a greeting message for a person",
        parameters=[
            ToolParameter(
                name="name",
                type="string",
                description="Name of person to greet",
                required=True,
            ),
            ToolParameter(
                name="greeting",
                type="string",
                description="Greeting to use (default: Hello)",
                required=False,
                default="Hello",
            ),
            ToolParameter(
                name="formal",
                type="boolean",
                description="Use formal greeting (default: false)",
                required=False,
                default=False,
            ),
        ],
    )

    # Register text analysis tool
    server.register_tool(
        name="analyze_text",
        function=analyze_text,
        description="Analyze text and return statistics",
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="Text to analyze",
                required=True,
            ),
            ToolParameter(
                name="include_words",
                type="boolean",
                description="Include word frequency analysis",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="include_sentences",
                type="boolean",
                description="Include sentence analysis",
                required=False,
                default=True,
            ),
        ],
    )

    # Register fetch tool
    server.register_tool(
        name="fetch_data",
        function=fetch_data,
        description="Fetch data from a URL",
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="URL to fetch data from",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Request timeout in seconds (1-300)",
                required=False,
                default=30,
            ),
        ],
    )

    logger.info("Registered 3 example tools")


# ============================================================================
# Testing
# ============================================================================


if __name__ == "__main__":
    # Test greet
    print("=" * 80)
    print("Testing greet tool")
    print("=" * 80)

    result = greet("Alice", formal=True)
    print(f"Result: {result}")

    # Test analyze_text
    print("\n" + "=" * 80)
    print("Testing analyze_text tool")
    print("=" * 80)

    text = "Hello world. This is a test. Testing analysis functionality."
    result = analyze_text(text)
    print(f"Result: {result}")

    # Test fetch_data
    print("\n" + "=" * 80)
    print("Testing fetch_data tool")
    print("=" * 80)

    result = fetch_data("https://api.example.com/data")
    print(f"Result: {result}")
