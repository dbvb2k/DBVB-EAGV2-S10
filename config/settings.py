"""
Centralized configuration management for the Multi-Agent System.

Loads configuration from YAML files and environment variables with validation.
"""
import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    model: str = Field(default="gemini-2.0-flash", description="Model name to use")
    api_key: Optional[str] = Field(default=None, description="API key (from env if None)")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    initial_retry_delay: float = Field(default=1.0, ge=0.1, description="Initial retry delay in seconds")
    max_retry_delay: float = Field(default=10.0, ge=1.0, description="Maximum retry delay in seconds")
    request_timeout: float = Field(default=60.0, ge=1.0, description="Request timeout in seconds")


class ExecutorConfig(BaseModel):
    """Code executor configuration settings."""
    max_functions: int = Field(default=5, ge=1, le=50, description="Maximum function calls allowed")
    timeout_per_function: float = Field(default=500.0, ge=1.0, description="Timeout per function call in seconds")
    min_timeout: float = Field(default=3.0, ge=1.0, description="Minimum timeout in seconds")
    max_ast_depth: int = Field(default=50, ge=10, le=200, description="Maximum AST depth")
    allowed_modules: List[str] = Field(
        default_factory=lambda: [
            "math", "cmath", "decimal", "fractions", "random", "statistics",
            "itertools", "functools", "operator", "string", "re", "datetime",
            "calendar", "time", "collections", "heapq", "bisect", "types",
            "copy", "enum", "uuid", "dataclasses", "typing", "pprint", "json",
            "base64", "hashlib", "hmac", "secrets", "struct", "zlib", "gzip",
            "bz2", "lzma", "io", "pathlib", "tempfile", "textwrap", "difflib",
            "unicodedata", "html", "html.parser", "xml", "xml.etree.ElementTree",
            "csv", "sqlite3", "contextlib", "traceback", "ast", "tokenize",
            "token", "builtins"
        ],
        description="List of allowed Python modules"
    )


class MemoryConfig(BaseModel):
    """Memory module configuration settings."""
    base_dir: str = Field(default="memory/session_logs", description="Base directory for session logs")
    search_top_k: int = Field(default=3, ge=1, le=20, description="Number of top search results")
    max_previous_failure_steps: int = Field(default=3, ge=1, le=10, description="Max failure steps to remember")
    concurrent_file_limit: int = Field(default=10, ge=1, le=50, description="Max concurrent file operations")


class AgentConfig(BaseModel):
    """Agent loop configuration settings."""
    max_iterations: int = Field(default=100, ge=10, le=1000, description="Maximum loop iterations")
    default_strategy: str = Field(default="exploratory", description="Default planning strategy")
    
    @field_validator('default_strategy')
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid_strategies = ["exploratory", "conservative"]
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    # Decision and Perception use same LLM config
    decision: LLMConfig = Field(default_factory=LLMConfig)
    perception: LLMConfig = Field(default_factory=LLMConfig)


class ConfigManager:
    """
    Centralized configuration manager.
    
    Loads configuration from YAML files and environment variables.
    Environment variables take precedence over YAML values.
    """
    
    def __init__(self, config_path: Optional[Path] = None, models_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML config file (defaults to config/profiles.yaml)
            models_path: Path to models.json file (defaults to config/models.json)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "profiles.yaml"
        if models_path is None:
            models_path = Path(__file__).parent / "models.json"
        
        self.config_path = config_path
        self.models_path = models_path
        self._models_config: Optional[Dict[str, Any]] = None
        self._config: Optional[AppConfig] = None
        self._load_models_config()
        self._load_config()
    
    def _load_models_config(self) -> None:
        """Load models.json configuration."""
        if self.models_path.exists():
            try:
                with open(self.models_path, 'r', encoding='utf-8') as f:
                    self._models_config = json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load models config from {self.models_path}: {e}")
                self._models_config = {}
        else:
            self._models_config = {}
    
    def _get_model_name(self, model_key: str) -> str:
        """
        Get actual model name from models.json using model key.
        
        Args:
            model_key: Model key from YAML (e.g., "gemini")
            
        Returns:
            Actual model name (e.g., "gemini-2.0-flash")
        """
        if not self._models_config or "models" not in self._models_config:
            # Fallback to default
            return "gemini-2.0-flash"
        
        models = self._models_config.get("models", {})
        if model_key in models:
            model_info = models[model_key]
            return model_info.get("model", "gemini-2.0-flash")
        
        # If key not found, assume it's already the model name
        return model_key
    
    def _load_config(self) -> None:
        """Load configuration from YAML and environment variables."""
        # Start with defaults
        config_dict = {}
        
        # Load from YAML if exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    config_dict = self._merge_yaml_config(yaml_config or {})
            except Exception as e:
                print(f"⚠️ Warning: Could not load config from {self.config_path}: {e}")
        
        # Override with environment variables
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create Pydantic model
        try:
            self._config = AppConfig(**config_dict)
        except Exception as e:
            print(f"⚠️ Warning: Config validation error: {e}, using defaults")
            self._config = AppConfig()
    
    def _merge_yaml_config(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML config into structured format."""
        config = {}
        
        # LLM config
        if "llm" in yaml_config:
            llm_config = yaml_config["llm"]
            model_key = llm_config.get("text_generation", "gemini")
            # Resolve actual model name from models.json
            actual_model = self._get_model_name(model_key)
            
            config["llm"] = {
                "model": actual_model,
                "api_key": os.getenv("GEMINI_API_KEY"),
            }
            config["decision"] = config["llm"].copy()
            config["perception"] = config["llm"].copy()
        
        # Agent config
        if "strategy" in yaml_config:
            strategy = yaml_config["strategy"]
            config["agent"] = {
                "default_strategy": strategy.get("planning_mode", "exploratory"),
                "max_iterations": strategy.get("max_steps", 100) * 10,  # Convert to iterations
            }
        
        # Memory config
        if "memory" in yaml_config:
            memory = yaml_config["memory"]
            if "storage" in memory:
                config["memory"] = {
                    "base_dir": memory["storage"].get("base_dir", "memory/session_logs"),
                }
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # LLM model
        if "GEMINI_MODEL" in os.environ:
            model = os.getenv("GEMINI_MODEL")
            if "llm" not in config:
                config["llm"] = {}
            config["llm"]["model"] = model
            if "decision" not in config:
                config["decision"] = {}
            config["decision"]["model"] = model
            if "perception" not in config:
                config["perception"] = {}
            config["perception"]["model"] = model
        
        # Retry settings
        if "LLM_MAX_RETRIES" in os.environ:
            retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
            for key in ["llm", "decision", "perception"]:
                if key not in config:
                    config[key] = {}
                config[key]["max_retries"] = retries
        
        # Timeout settings
        if "LLM_TIMEOUT" in os.environ:
            timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))
            for key in ["llm", "decision", "perception"]:
                if key not in config:
                    config[key] = {}
                config[key]["request_timeout"] = timeout
        
        # Executor settings
        if "EXECUTOR_MAX_FUNCTIONS" in os.environ:
            if "executor" not in config:
                config["executor"] = {}
            config["executor"]["max_functions"] = int(os.getenv("EXECUTOR_MAX_FUNCTIONS", "5"))
        
        if "EXECUTOR_TIMEOUT" in os.environ:
            if "executor" not in config:
                config["executor"] = {}
            config["executor"]["timeout_per_function"] = float(os.getenv("EXECUTOR_TIMEOUT", "500.0"))
        
        # Agent settings
        if "AGENT_MAX_ITERATIONS" in os.environ:
            if "agent" not in config:
                config["agent"] = {}
            config["agent"]["max_iterations"] = int(os.getenv("AGENT_MAX_ITERATIONS", "100"))
        
        return config
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def reload(self) -> None:
        """Reload configuration from files and environment."""
        self._load_config()
    
    def get_llm_config(self, component: str = "llm") -> LLMConfig:
        """Get LLM configuration for a specific component."""
        config = self.config
        if component == "decision":
            return config.decision
        elif component == "perception":
            return config.perception
        else:
            return config.llm
    
    def get_executor_config(self) -> ExecutorConfig:
        """Get executor configuration."""
        return self.config.executor
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration."""
        return self.config.memory
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration."""
        return self.config.agent


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config(config_path: Optional[Path] = None) -> ConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def reload_config() -> None:
    """Reload global configuration."""
    global _config_manager
    if _config_manager is not None:
        _config_manager.reload()

