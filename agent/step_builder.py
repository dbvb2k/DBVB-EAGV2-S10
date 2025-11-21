"""
StepBuilder - Builder pattern for creating Step objects with validation.

This module provides a fluent builder interface for constructing Step objects,
ensuring proper validation and type safety throughout the construction process.
"""

from datetime import datetime
from typing import Optional, List, Any
from agent.agentSession import Step, StepType, StepStatus, ToolCode, PerceptionSnapshot


class StepBuilder:
    """
    Builder for creating Step objects with fluent interface and validation.
    
    Example usage:
        step = (StepBuilder()
            .with_index(0)
            .with_description("Search documents")
            .with_type(StepType.CODE)
            .with_code("result = search_documents(query)")
            .with_dependency(1)
            .build())
    """
    
    def __init__(self):
        """Initialize a new StepBuilder."""
        self._index: Optional[int] = None
        self._description: Optional[str] = None
        self._type: Optional[StepType] = None
        self._status: StepStatus = StepStatus.PENDING
        self._code: Optional[ToolCode] = None
        self._conclusion: Optional[str] = None
        self._execution_result: Optional[Any] = None
        self._error: Optional[str] = None
        self._perception: Optional[PerceptionSnapshot] = None
        self._attempts: int = 0
        self._was_replanned: bool = False
        self._parent_index: Optional[int] = None
        self._dependencies: List[int] = []
        self._created_at: Optional[datetime] = None
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
    
    def with_index(self, index: int) -> 'StepBuilder':
        """Set the step index."""
        if index < 0:
            raise ValueError(f"Step index must be >= 0, got {index}")
        self._index = index
        return self
    
    def with_description(self, description: str) -> 'StepBuilder':
        """Set the step description."""
        if not description or not description.strip():
            raise ValueError("Step description cannot be empty")
        self._description = description.strip()
        return self
    
    def with_type(self, step_type: StepType) -> 'StepBuilder':
        """Set the step type."""
        if not isinstance(step_type, StepType):
            raise TypeError(f"step_type must be StepType, got {type(step_type)}")
        self._type = step_type
        return self
    
    def with_status(self, status: StepStatus) -> 'StepBuilder':
        """Set the step status."""
        if not isinstance(status, StepStatus):
            raise TypeError(f"status must be StepStatus, got {type(status)}")
        self._status = status
        return self
    
    def with_code(self, code: str, tool_name: str = "raw_code_block") -> 'StepBuilder':
        """
        Set the code for a CODE step.
        
        Args:
            code: The code string to execute
            tool_name: Optional tool name (default: "raw_code_block")
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty for CODE step")
        self._code = ToolCode(
            tool_name=tool_name,
            tool_arguments={"code": code.strip()}
        )
        return self
    
    def with_tool_code(self, tool_code: ToolCode) -> 'StepBuilder':
        """Set the ToolCode object directly."""
        if not isinstance(tool_code, ToolCode):
            raise TypeError(f"tool_code must be ToolCode, got {type(tool_code)}")
        self._code = tool_code
        return self
    
    def with_conclusion(self, conclusion: str) -> 'StepBuilder':
        """Set the conclusion for a CONCLUDE step."""
        if conclusion is None:
            self._conclusion = None
            return self
        if not conclusion or not conclusion.strip():
            # Allow empty conclusion during building - will be validated in build() if type is CONCLUDE
            self._conclusion = None
            return self
        self._conclusion = conclusion.strip()
        return self
    
    def with_execution_result(self, result: Any) -> 'StepBuilder':
        """Set the execution result."""
        self._execution_result = result
        return self
    
    def with_error(self, error: str) -> 'StepBuilder':
        """Set the error message."""
        self._error = error
        return self
    
    def with_perception(self, perception: PerceptionSnapshot) -> 'StepBuilder':
        """Set the perception snapshot."""
        if not isinstance(perception, PerceptionSnapshot):
            raise TypeError(f"perception must be PerceptionSnapshot, got {type(perception)}")
        self._perception = perception
        return self
    
    def with_attempts(self, attempts: int) -> 'StepBuilder':
        """Set the number of attempts."""
        if attempts < 0:
            raise ValueError(f"Attempts must be >= 0, got {attempts}")
        self._attempts = attempts
        return self
    
    def with_was_replanned(self, was_replanned: bool) -> 'StepBuilder':
        """Set whether the step was replanned."""
        self._was_replanned = was_replanned
        return self
    
    def with_parent_index(self, parent_index: Optional[int]) -> 'StepBuilder':
        """Set the parent step index (if replanned)."""
        if parent_index is not None and parent_index < 0:
            raise ValueError(f"Parent index must be >= 0, got {parent_index}")
        self._parent_index = parent_index
        return self
    
    def with_dependency(self, step_index: int) -> 'StepBuilder':
        """Add a dependency on another step."""
        if step_index < 0:
            raise ValueError(f"Dependency step index must be >= 0, got {step_index}")
        if step_index not in self._dependencies:
            self._dependencies.append(step_index)
        return self
    
    def with_dependencies(self, dependencies: List[int]) -> 'StepBuilder':
        """Set multiple dependencies."""
        for dep in dependencies:
            if dep < 0:
                raise ValueError(f"Dependency step index must be >= 0, got {dep}")
        self._dependencies = list(set(dependencies))  # Remove duplicates
        return self
    
    def with_created_at(self, created_at: datetime) -> 'StepBuilder':
        """Set the creation timestamp."""
        if not isinstance(created_at, datetime):
            raise TypeError(f"created_at must be datetime, got {type(created_at)}")
        self._created_at = created_at
        return self
    
    def with_started_at(self, started_at: Optional[datetime]) -> 'StepBuilder':
        """Set the started timestamp."""
        if started_at is not None and not isinstance(started_at, datetime):
            raise TypeError(f"started_at must be datetime or None, got {type(started_at)}")
        self._started_at = started_at
        return self
    
    def with_completed_at(self, completed_at: Optional[datetime]) -> 'StepBuilder':
        """Set the completed timestamp."""
        if completed_at is not None and not isinstance(completed_at, datetime):
            raise TypeError(f"completed_at must be datetime or None, got {type(completed_at)}")
        self._completed_at = completed_at
        return self
    
    def from_decision_output(self, decision_output: dict) -> 'StepBuilder':
        """
        Initialize builder from decision module output.
        
        Args:
            decision_output: Dictionary from decision module with keys:
                - step_index, description, type, code, conclusion, etc.
        """
        if "step_index" in decision_output:
            self.with_index(decision_output["step_index"])
        if "description" in decision_output:
            self.with_description(decision_output["description"])
        if "type" in decision_output:
            # Handle string or enum
            step_type = decision_output["type"]
            if isinstance(step_type, str):
                step_type = StepType[step_type.upper()] if step_type.upper() in StepType.__members__ else StepType.NOP
            self.with_type(step_type)
        if "code" in decision_output and decision_output["code"]:
            code_str = decision_output["code"]
            tool_name = decision_output.get("tool_name", "raw_code_block")
            self.with_code(code_str, tool_name)
        # Only set conclusion if it exists and is non-empty
        # Note: We don't validate here - validation happens in build()
        if "conclusion" in decision_output and decision_output["conclusion"]:
            # Store conclusion without validation (will be validated in build() if type is CONCLUDE)
            self._conclusion = decision_output["conclusion"].strip()
        if "status" in decision_output:
            status = decision_output["status"]
            if isinstance(status, str):
                try:
                    status = StepStatus[status.upper()]
                except KeyError:
                    status = StepStatus.PENDING
            self.with_status(status)
        if "dependencies" in decision_output:
            self.with_dependencies(decision_output["dependencies"])
        if "parent_index" in decision_output:
            self.with_parent_index(decision_output["parent_index"])
        if "was_replanned" in decision_output:
            self.with_was_replanned(decision_output["was_replanned"])
        
        return self
    
    def build(self) -> Step:
        """
        Build the Step object with validation.
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if self._index is None:
            raise ValueError("Step index is required")
        if not self._description:
            raise ValueError("Step description is required")
        if self._type is None:
            raise ValueError("Step type is required")
        
        # Type-specific validation
        if self._type == StepType.CODE and self._code is None:
            raise ValueError("CODE step type requires code")
        if self._type == StepType.CONCLUDE:
            if not self._conclusion or not self._conclusion.strip():
                raise ValueError("CONCLUDE step type requires non-empty conclusion")
        
        # Build the step
        step_data = {
            "index": self._index,
            "description": self._description,
            "type": self._type,
            "status": self._status,
            "code": self._code,
            "conclusion": self._conclusion,
            "execution_result": self._execution_result,
            "error": self._error,
            "perception": self._perception,
            "attempts": self._attempts,
            "was_replanned": self._was_replanned,
            "parent_index": self._parent_index,
            "dependencies": self._dependencies,
        }
        
        # Add timestamps if provided
        if self._created_at:
            step_data["created_at"] = self._created_at
        if self._started_at:
            step_data["started_at"] = self._started_at
        if self._completed_at:
            step_data["completed_at"] = self._completed_at
        
        return Step(**step_data)
    
    def reset(self) -> 'StepBuilder':
        """Reset the builder to initial state."""
        self.__init__()
        return self

