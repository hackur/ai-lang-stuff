"""
Filesystem MCP Server

Provides safe filesystem operations through MCP protocol.
Operations are restricted to allowed directories for security.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FilesystemMCPServer:
    """
    MCP server for filesystem operations.

    Provides read, write, list, and search operations with security restrictions.
    """

    def __init__(self, allowed_paths: Optional[List[str]] = None):
        """
        Initialize filesystem MCP server.

        Args:
            allowed_paths: List of directory paths where operations are allowed.
                          If None, only current directory and subdirectories allowed.
        """
        if allowed_paths is None:
            self.allowed_paths = [Path.cwd()]
        else:
            self.allowed_paths = [Path(p).resolve() for p in allowed_paths]

        logger.info(f"Filesystem MCP server initialized with {len(self.allowed_paths)} allowed paths")

    def _is_path_allowed(self, path: Union[str, Path]) -> bool:
        """Check if path is within allowed directories."""
        resolved_path = Path(path).resolve()

        for allowed in self.allowed_paths:
            try:
                resolved_path.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False

    def read_file(self, path: str, encoding: str = "utf-8") -> Dict[str, any]:
        """
        Read file contents.

        Args:
            path: Path to file
            encoding: File encoding

        Returns:
            Dict with status and content or error

        Example:
            >>> server = FilesystemMCPServer()
            >>> result = server.read_file("README.md")
            >>> if result["status"] == "success":
            ...     print(result["content"])
        """
        try:
            file_path = Path(path)

            if not self._is_path_allowed(file_path):
                return {
                    "status": "error",
                    "error": f"Access denied: {path} is outside allowed paths",
                }

            if not file_path.exists():
                return {"status": "error", "error": f"File not found: {path}"}

            if not file_path.is_file():
                return {"status": "error", "error": f"Not a file: {path}"}

            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            logger.info(f"Read file: {path} ({len(content)} chars)")

            return {
                "status": "success",
                "path": str(file_path),
                "content": content,
                "size": len(content),
            }

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return {"status": "error", "error": str(e)}

    def write_file(self, path: str, content: str, encoding: str = "utf-8") -> Dict[str, any]:
        """
        Write content to file.

        Args:
            path: Path to file
            content: Content to write
            encoding: File encoding

        Returns:
            Dict with status

        Example:
            >>> server = FilesystemMCPServer()
            >>> result = server.write_file("output.txt", "Hello World")
            >>> print(result["status"])
            success
        """
        try:
            file_path = Path(path)

            if not self._is_path_allowed(file_path):
                return {
                    "status": "error",
                    "error": f"Access denied: {path} is outside allowed paths",
                }

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)

            logger.info(f"Wrote file: {path} ({len(content)} chars)")

            return {
                "status": "success",
                "path": str(file_path),
                "bytes_written": len(content),
            }

        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return {"status": "error", "error": str(e)}

    def list_directory(self, path: str, pattern: Optional[str] = None) -> Dict[str, any]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Optional glob pattern (e.g., "*.py")

        Returns:
            Dict with status and list of files/directories

        Example:
            >>> server = FilesystemMCPServer()
            >>> result = server.list_directory(".", pattern="*.py")
            >>> print(result["files"])
            ['main.py', 'test.py']
        """
        try:
            dir_path = Path(path)

            if not self._is_path_allowed(dir_path):
                return {
                    "status": "error",
                    "error": f"Access denied: {path} is outside allowed paths",
                }

            if not dir_path.exists():
                return {"status": "error", "error": f"Directory not found: {path}"}

            if not dir_path.is_dir():
                return {"status": "error", "error": f"Not a directory: {path}"}

            # List contents
            if pattern:
                items = list(dir_path.glob(pattern))
            else:
                items = list(dir_path.iterdir())

            files = []
            directories = []

            for item in sorted(items):
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "path": str(item),
                        "size": item.stat().st_size,
                    })
                elif item.is_dir():
                    directories.append({
                        "name": item.name,
                        "path": str(item),
                    })

            logger.info(f"Listed directory: {path} ({len(files)} files, {len(directories)} dirs)")

            return {
                "status": "success",
                "path": str(dir_path),
                "files": files,
                "directories": directories,
                "total_items": len(files) + len(directories),
            }

        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return {"status": "error", "error": str(e)}

    def search_files(self, path: str, query: str, file_pattern: str = "**/*") -> Dict[str, any]:
        """
        Search for files containing text.

        Args:
            path: Root directory to search
            query: Text to search for
            file_pattern: Glob pattern for files to search

        Returns:
            Dict with status and matching files

        Example:
            >>> server = FilesystemMCPServer()
            >>> result = server.search_files(".", "TODO", "**/*.py")
            >>> for match in result["matches"]:
            ...     print(f"{match['file']}: {match['line']}")
        """
        try:
            root_path = Path(path)

            if not self._is_path_allowed(root_path):
                return {
                    "status": "error",
                    "error": f"Access denied: {path} is outside allowed paths",
                }

            if not root_path.exists() or not root_path.is_dir():
                return {"status": "error", "error": f"Invalid directory: {path}"}

            matches = []

            for file_path in root_path.glob(file_pattern):
                if not file_path.is_file():
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                matches.append({
                                    "file": str(file_path),
                                    "line_number": line_num,
                                    "line": line.strip(),
                                })
                except Exception as e:
                    logger.debug(f"Error reading {file_path}: {e}")
                    continue

            logger.info(f"Search in {path} for '{query}': {len(matches)} matches")

            return {
                "status": "success",
                "query": query,
                "matches": matches,
                "total_matches": len(matches),
            }

        except Exception as e:
            logger.error(f"Error searching files in {path}: {e}")
            return {"status": "error", "error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging(level="DEBUG")

    # Create server
    server = FilesystemMCPServer(allowed_paths=["."])

    # Test read
    result = server.read_file("README.md")
    print(json.dumps(result, indent=2))

    # Test list
    result = server.list_directory(".", pattern="*.md")
    print(json.dumps(result, indent=2))

    # Test search
    result = server.search_files(".", "LangChain", "**/*.md")
    print(json.dumps(result, indent=2))
