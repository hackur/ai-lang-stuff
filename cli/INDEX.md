# ailang CLI - Documentation Index

Quick navigation to all CLI documentation.

## Getting Started

Start here if you're new to ailang:

1. **[README.md](README.md)** - Full user documentation (7.9K)
   - Overview and features
   - Installation instructions
   - Usage examples
   - Troubleshooting guide

2. **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide (5.6K)
   - 5-minute setup
   - First steps
   - Common workflows
   - Tips and tricks

## Reference Documentation

Detailed information about commands and features:

3. **[COMMANDS.md](COMMANDS.md)** - Complete command reference (16K)
   - All commands with syntax
   - Options and arguments
   - Usage examples
   - Command chaining

4. **[FEATURES.md](FEATURES.md)** - Feature overview (9.7K)
   - Core features
   - UI/UX capabilities
   - Advanced features
   - Roadmap

## Technical Documentation

For developers and contributors:

5. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture (14K)
   - Design principles
   - Component details
   - Data flow
   - Extension points

6. **[STATUS.md](STATUS.md)** - Implementation status (10K)
   - Checklist of completed features
   - Statistics and metrics
   - Known limitations
   - Future enhancements

## Installation & Setup

### Quick Install
```bash
cd /Volumes/JS-DEV/ai-lang-stuff/cli
./install.sh
```

### Manual Install
```bash
pip install -e .
```

### Verify
```bash
ailang --version
ailang --help
```

## Key Commands by Use Case

### Model Management
```bash
ailang models recommend coding    # Get recommendations
ailang models pull qwen3:8b      # Download model
ailang models benchmark qwen3:8b # Test performance
```

### Example Runner
```bash
ailang examples list             # Browse examples
ailang examples run <name>       # Execute example
ailang examples validate         # Test all examples
```

### MCP Servers
```bash
ailang mcp list                  # Show servers
ailang mcp start <name> --background  # Start server
ailang mcp stop <name>           # Stop server
```

### RAG System
```bash
ailang rag index ./docs --name my-docs  # Index documents
ailang rag query my-docs "query"        # Search
ailang rag list                         # Show collections
```

## File Structure

```
cli/
├── ailang/                    # Main package
│   ├── __init__.py
│   ├── main.py               # Entry point
│   └── commands/             # Command modules
│       ├── models.py         # 288 lines
│       ├── examples.py       # 235 lines
│       ├── mcp.py           # 310 lines
│       └── rag.py           # 311 lines
├── tests/                    # Test suite
│   ├── test_cli.py
│   └── test_commands.py
├── Documentation (this folder)
│   ├── README.md            # Start here
│   ├── QUICKSTART.md        # Quick setup
│   ├── COMMANDS.md          # Command reference
│   ├── FEATURES.md          # Feature list
│   ├── ARCHITECTURE.md      # Technical docs
│   ├── STATUS.md            # Status report
│   └── INDEX.md             # This file
├── Scripts
│   ├── install.sh           # Installation
│   ├── demo.sh             # Demo
│   └── Makefile            # Dev commands
└── Configuration
    ├── setup.py            # Legacy install
    ├── pyproject.toml     # Modern packaging
    └── .gitignore         # Git exclusions
```

## Documentation by Topic

### Installation
- README.md § Installation
- QUICKSTART.md § Installation
- install.sh (script)

### Command Usage
- COMMANDS.md (complete reference)
- README.md § Commands
- QUICKSTART.md § Usage Examples

### Features
- FEATURES.md (comprehensive list)
- README.md § Features
- STATUS.md § Features Implemented

### Architecture
- ARCHITECTURE.md (technical details)
- STATUS.md § Project Structure
- README.md § Directory Structure

### Development
- ARCHITECTURE.md § Extension Points
- Makefile (dev commands)
- tests/ (test suite)

### Troubleshooting
- README.md § Troubleshooting
- QUICKSTART.md § Troubleshooting
- COMMANDS.md § Error Handling

## Quick Links

**I want to...**

- **Install the CLI** → [QUICKSTART.md](QUICKSTART.md) or [install.sh](install.sh)
- **Learn basic usage** → [QUICKSTART.md](QUICKSTART.md) § First Steps
- **See all commands** → [COMMANDS.md](COMMANDS.md)
- **Understand features** → [FEATURES.md](FEATURES.md)
- **Contribute code** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Check progress** → [STATUS.md](STATUS.md)
- **Get help** → [README.md](README.md) § Getting Help

## External Resources

- **Project Repository**: GitHub (see main README)
- **Issue Tracker**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Click Framework**: https://click.palletsprojects.com/
- **Rich Library**: https://rich.readthedocs.io/

## Statistics

- **Total Documentation**: 60K+ characters
- **Total Commands**: 16+
- **Total Code**: 1,145 lines
- **Total Files**: 21
- **Test Files**: 2

## Version

- **Current Version**: 0.1.0
- **Status**: Production-Ready
- **Last Updated**: October 26, 2024

## Contributing

To contribute to documentation:

1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Follow existing documentation style
3. Update INDEX.md when adding new docs
4. Keep cross-references up to date
5. Test all examples and commands

## Support

For help:
1. Check relevant documentation above
2. Use `ailang --help` or `ailang <command> --help`
3. Review examples in QUICKSTART.md
4. Submit issues on GitHub

---

**Navigation Tip**: Use Ctrl+F / Cmd+F to search this index for specific topics.
