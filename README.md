# Agentic Query Assistant - Multi-Agent System Framework

A sophisticated multi-agent AI framework that uses perception, decision-making, and action execution to solve complex queries through an iterative, goal-oriented approach. Built with Python 3.12+ and designed for extensibility and production use.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Improvements Summary](#improvements-summary)

## Overview

The Agentic Query Assistant is an intelligent system that breaks down complex queries into manageable steps, executes them using available tools, and iteratively refines its approach based on perception and feedback. The framework implements a complete agent loop with:

- **Perception**: Analyzes inputs and determines goal achievement status
- **Decision**: Generates execution plans and strategies
- **Action**: Executes code/tools in a sandboxed environment
- **Memory**: Searches and stores session history for context
- **State Management**: Tracks session lifecycle with proper state machines

## Features

### Core Capabilities
- 🔄 **Iterative Planning**: Dynamic plan generation and replanning based on execution results
- 🧠 **Memory System**: Semantic search over past sessions for context and learning
- 🛠️ **Tool Integration**: Seamless integration with MCP (Model Context Protocol) servers
- 🔒 **Sandboxed Execution**: Secure code execution with resource limits and validation
- 📊 **Structured Logging**: Comprehensive logging with correlation IDs and structured output
- ✅ **Type Safety**: Pydantic models for validation and type safety throughout
- ⚙️ **Configuration Management**: Centralized, environment-aware configuration

### Production-Ready Features
- Async/await throughout for non-blocking operations
- Retry logic with exponential backoff for LLM calls
- Error handling and recovery mechanisms
- Session state management with lifecycle tracking
- Observer pattern for extensible notifications
- Event logging for audit trails

## Architecture

The framework follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Loop                            │
│  (Orchestrates perception, decision, action cycles)    │
└─────────────────────────────────────────────────────────┘
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │Perception│    │ Decision │    │  Action  │
    │          │    │          │    │          │
    │ Analyzes │    │ Plans    │    │ Executes │
    │ Context  │    │ Steps    │    │ Code     │
    └──────────┘    └──────────┘    └──────────┘
           │              │              │
           └──────────────┴──────────────┘
                         │
                         ▼
                  ┌──────────┐
                  │  Memory  │
                  │          │
                  │  Stores  │
                  │  &       │
                  │  Searches│
                  └──────────┘
```

### Component Flow

1. **User Query** → Agent Loop receives query
2. **Memory Search** → Searches past sessions for relevant context
3. **Perception** → Analyzes query and determines if goal can be achieved
4. **Decision** → Generates execution plan with steps
5. **Action** → Executes steps using available tools
6. **Evaluation** → Perceives results and decides next action
7. **Iteration** → Repeats until goal achieved or max iterations reached

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Google Gemini API key (for LLM access)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Session-10
   ```

2. **Install dependencies**
   
   Using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Configure MCP servers**
   
   Edit `config/mcp_server_config.yaml` to configure your MCP servers.

## Configuration

The framework uses a centralized configuration system with YAML files and environment variable overrides.

### Configuration Files

- `config/profiles.yaml` - Main configuration file
- `config/mcp_server_config.yaml` - MCP server configurations
- `config/settings.py` - Configuration management module

### Key Configuration Options

**LLM Settings** (`config/profiles.yaml`):
```yaml
llm:
  model: gemini-2.0-flash
  max_retries: 3
  request_timeout: 60.0
```

**Executor Settings**:
```yaml
executor:
  max_functions: 5
  timeout_per_function: 500.0
  max_ast_depth: 50
```

**Agent Settings**:
```yaml
agent:
  max_iterations: 100
  default_strategy: exploratory
```

**Memory Settings**:
```yaml
memory:
  base_dir: memory/session_logs
  search_top_k: 3
  max_previous_failure_steps: 3
```

### Environment Variables

You can override configuration using environment variables:

- `GEMINI_MODEL` - Override LLM model name
- `LLM_MAX_RETRIES` - Override retry count
- `LLM_TIMEOUT` - Override timeout
- `EXECUTOR_MAX_FUNCTIONS` - Override function limit
- `EXECUTOR_TIMEOUT` - Override executor timeout
- `AGENT_MAX_ITERATIONS` - Override iteration limit

## Usage

### Interactive Mode

Run the interactive assistant:

```bash
uv run main.py
```

Or with Python:

```bash
python main.py
```

### Programmatic Usage

```python
import asyncio
from agent.agent_loop import AgentLoop
from mcp_servers.multiMCP import MultiMCP

async def main():
    # Initialize MCP servers
    multi_mcp = MultiMCP(server_configs=[...])
    await multi_mcp.initialize()
    
    # Create agent loop
    loop = AgentLoop(
        perception_prompt_path="prompts/perception_prompt.txt",
        decision_prompt_path="prompts/decision_prompt.txt",
        multi_mcp=multi_mcp,
        strategy="exploratory"
    )
    
    # Run query
    session = await loop.run("Your query here")
    
    # Access results
    print(session.state.final_answer)
    print(session.state.solution_summary)

asyncio.run(main())
```

## Project Structure

```
Session-10/
├── action/                 # Code execution module
│   └── executor.py        # Sandboxed code executor
├── agent/                 # Agent orchestration
│   ├── agent_loop.py      # Main agent loop
│   ├── agentSession.py   # Session management
│   ├── step_builder.py   # Step construction (Builder pattern)
│   └── session_state_machine.py  # Lifecycle state machine
├── config/                # Configuration management
│   ├── settings.py        # Centralized config
│   ├── profiles.yaml     # Configuration profiles
│   └── mcp_server_config.yaml  # MCP server config
├── decision/              # Decision/planning module
│   └── decision.py       # Plan generation
├── memory/                # Memory system
│   ├── memory_search.py  # Semantic search
│   └── session_log.py    # Session persistence
├── perception/            # Perception module
│   └── perception.py     # Goal achievement analysis
├── models/                # Data models
│   └── schemas.py        # Pydantic validation models
├── mcp_servers/           # MCP server implementations
│   ├── multiMCP.py       # MCP dispatcher
│   └── documents/        # Document storage
├── prompts/               # LLM prompts
│   ├── perception_prompt.txt
│   └── decision_prompt.txt
├── utils/                 # Utilities
│   └── logger.py         # Structured logging
└── main.py               # Entry point
```

## Key Components

### Agent Loop (`agent/agent_loop.py`)
Orchestrates the complete agent lifecycle:
- Memory search
- Perception analysis
- Decision/planning
- Step execution
- Result evaluation
- Replanning when needed

### Perception Module (`perception/perception.py`)
Analyzes inputs and execution results to determine:
- Goal achievement status
- Confidence levels
- Required next steps
- Solution summaries

### Decision Module (`decision/decision.py`)
Generates execution plans:
- Step-by-step plans
- Tool selection
- Strategy selection (exploratory/conservative)
- Plan extension and replanning

### Action Executor (`action/executor.py`)
Executes user code safely:
- AST validation
- Sandboxed execution environment
- Resource limits
- Error handling

### Memory System (`memory/`)
- **Memory Search**: Semantic search over past sessions
- **Session Logging**: Persistent storage of session data
- Async file operations for performance

### Session Management (`agent/agentSession.py`)
- Immutable Pydantic models
- Observer pattern for notifications
- Event logging
- State machine for lifecycle management

## Development

### Setting Up Development Environment

1. Install development dependencies:
   ```bash
   uv sync --dev
   ```

2. Set up pre-commit hooks (if configured):
   ```bash
   pre-commit install
   ```

### Code Style

The project follows Python best practices:
- Type hints throughout
- Pydantic models for validation
- Async/await for I/O operations
- Structured logging
- Comprehensive error handling

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_decision.py

# Run with coverage
pytest --cov=.
```

## Testing

### Unit Tests

Test files are located alongside modules:
- `decision/decision_test.py`
- `perception/perception_test.py`

### Integration Tests

Test the complete agent loop with various scenarios:
- Simple queries
- Complex multi-step tasks
- Error recovery
- Replanning scenarios

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following the code style
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Development Guidelines

- Write clear, descriptive commit messages
- Add type hints to all functions
- Include docstrings for public APIs
- Update relevant documentation
- Ensure all tests pass

## License

[Specify your license here]

## Improvements Summary

<!-- TODO: Add summary of improvements made to the framework -->

This section will contain a comprehensive summary of all improvements, enhancements, and refactoring work completed on the framework. See the following documentation files for detailed information:

- `FINAL_IMPROVEMENTS_SUMMARY.md` - Latest improvements (logging, validation, security)
- `IMPROVEMENTS_SUMMARY_BY_PACKAGE.md` - Detailed improvements organized by package
- `VERIFICATION_REPORT.md` - Verification of recommendations implementation
- `FIXES_COMPLETED.md` - Critical fixes implementation summary
- `IMPLEMENTATION_COMPLETE.md` - Complete implementation status
- `FRAMEWORK_ANALYSIS.md` - Original framework analysis
- `AGENT_SESSION_ARCHITECTURE_ANALYSIS.md` - Architecture analysis and recommendations

---

## Support

For issues, questions, or contributions, please open an issue on the repository.

**Last Updated**: 2025-01-08

