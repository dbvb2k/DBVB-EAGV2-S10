"""
Agent Session Models - Pydantic Models for Type Safety and Validation

This module provides Pydantic models for agent session management:
- ToolCode: Immutable tool invocation metadata
- PerceptionSnapshot: Immutable perception analysis snapshot with confidence validation
- Step: Enhanced step with status transitions and dependency tracking
- PlanVersion: Plan version with metadata and versioning support
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import Any, List, Optional, Dict, Protocol
from enum import Enum
from datetime import datetime
from decimal import Decimal
import json
import time


def _convert_decimals_to_str(obj: Any) -> Any:
    """Recursively convert Decimal values to strings for JSON serialization."""
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimals_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_decimals_to_str(item) for item in obj)
    else:
        return obj


class StepType(str, Enum):
    """Valid step types."""
    CODE = "CODE"
    CONCLUDE = "CONCLUDE"
    NOP = "NOP"
    NOOP = "NOOP"  # Alias for NOP for backward compatibility


class StepStatus(str, Enum):
    """Valid step statuses."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CLARIFICATION_NEEDED = "clarification_needed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ToolCode(BaseModel):
    """Immutable tool code representation."""
    tool_name: str = Field(min_length=1, description="Tool name")
    tool_arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    
    model_config = ConfigDict(frozen=True)  # Immutable in Pydantic v2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility)."""
        return {
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments
        }


class PerceptionSnapshot(BaseModel):
    """Immutable perception snapshot with confidence validation."""
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    result_requirement: str = Field(default="", description="Result requirement description")
    original_goal_achieved: bool = Field(default=False, description="Whether original goal is achieved")
    reasoning: str = Field(default="", description="Reasoning about goal achievement")
    local_goal_achieved: bool = Field(default=False, description="Whether local/step goal is achieved")
    local_reasoning: str = Field(default="", description="Reasoning about local goal")
    last_tooluse_summary: str = Field(default="", description="Summary of last tool use")
    solution_summary: str = Field(default="", description="Solution summary")
    confidence: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Confidence score (0.0-1.0)"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    model_config = ConfigDict(
        frozen=True,  # Immutable in Pydantic v2
        json_encoders={Decimal: str, datetime: lambda v: v.isoformat()}
    )
    
    @field_validator('confidence', mode='before')
    @classmethod
    def parse_confidence(cls, v: Any) -> Decimal:
        """Parse and validate confidence value."""
        if isinstance(v, str):
            try:
                conf_decimal = Decimal(v)
            except (ValueError, TypeError):
                return Decimal("0.0")
        elif isinstance(v, (int, float)):
            conf_decimal = Decimal(str(v))
        elif isinstance(v, Decimal):
            conf_decimal = v
        else:
            return Decimal("0.0")
        
        # Clamp to [0.0, 1.0]
        return max(Decimal("0.0"), min(Decimal("1.0"), conf_decimal))
    
    @field_validator('entities', mode='before')
    @classmethod
    def validate_entities(cls, v: Any) -> List[str]:
        """Ensure entities is a list of strings."""
        if not isinstance(v, list):
            return []
        return [str(e) for e in v]
    
    @field_validator('original_goal_achieved', 'local_goal_achieved', mode='before')
    @classmethod
    def normalize_bool(cls, v: Any) -> bool:
        """Normalize boolean values from various formats."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)


class Step(BaseModel):
    """Enhanced step in agent execution with full validation and status transitions."""
    index: int = Field(ge=0, description="Step index")
    description: str = Field(min_length=1, description="Step description")
    type: StepType = Field(description="Step type")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Step status")
    
    # Type-specific fields
    code: Optional[ToolCode] = None
    conclusion: Optional[str] = None
    
    # Execution results
    execution_result: Optional[Any] = None
    error: Optional[str] = None
    perception: Optional[PerceptionSnapshot] = None
    
    # Metadata and tracking
    attempts: int = Field(default=0, ge=0, description="Number of attempts")
    was_replanned: bool = Field(default=False, description="Whether step was replanned")
    parent_index: Optional[int] = Field(default=None, ge=0, description="Parent step index if replanned")
    dependencies: List[int] = Field(default_factory=list, description="Step dependencies (list of step indices)")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    @field_validator('type', mode='before')
    @classmethod
    def normalize_step_type(cls, v: Any) -> StepType:
        """Normalize step type to enum."""
        if isinstance(v, str):
            v = v.upper()
            # Handle NOOP alias for backward compatibility
            if v == "NOOP":
                v = "NOP"
            try:
                return StepType[v]
            except KeyError:
                raise ValueError(f"Invalid step type: {v}")
        return v
    
    @field_validator('status', mode='before')
    @classmethod
    def normalize_status(cls, v: Any) -> StepStatus:
        """Normalize status to enum."""
        if isinstance(v, str):
            try:
                return StepStatus[v.upper()]
            except KeyError:
                # Try lowercase
                try:
                    return StepStatus(v.lower())
                except ValueError:
                    return StepStatus.PENDING
        return v
    
    @field_validator('code')
    @classmethod
    def validate_code_for_type(cls, v: Optional[ToolCode], info) -> Optional[ToolCode]:
        """Validate code is present for CODE type steps."""
        step_type = info.data.get('type')
        if step_type == StepType.CODE and v is None:
            raise ValueError("CODE step type requires code")
        return v
    
    @field_validator('conclusion')
    @classmethod
    def validate_conclusion_for_type(cls, v: Optional[str], info) -> Optional[str]:
        """Validate conclusion is present for CONCLUDE type steps."""
        step_type = info.data.get('type')
        if step_type == StepType.CONCLUDE and (v is None or not v.strip()):
            raise ValueError("CONCLUDE step type requires non-empty conclusion")
        return v
    
    def mark_started(self) -> 'Step':
        """Create new step with started_at timestamp and RETRYING status."""
        return self.model_copy(update={
            'started_at': datetime.now(),
            'status': StepStatus.RETRYING if self.attempts > 0 else self.status
        })
    
    def mark_completed(self, result: Any, perception: Optional[PerceptionSnapshot] = None) -> 'Step':
        """Create new step with completion data."""
        return self.model_copy(update={
            'status': StepStatus.COMPLETED,
            'completed_at': datetime.now(),
            'execution_result': result,
            'perception': perception
        })
    
    def mark_failed(self, error: str) -> 'Step':
        """Create new step with failure data."""
        return self.model_copy(update={
            'status': StepStatus.FAILED,
            'completed_at': datetime.now(),
            'error': error,
            'attempts': self.attempts + 1
        })
    
    def mark_skipped(self, reason: Optional[str] = None) -> 'Step':
        """Create new step with skipped status."""
        return self.model_copy(update={
            'status': StepStatus.SKIPPED,
            'completed_at': datetime.now(),
            'error': reason
        })
    
    def add_dependency(self, step_index: int) -> 'Step':
        """Add a dependency on another step."""
        if step_index not in self.dependencies:
            new_deps = list(self.dependencies) + [step_index]
            return self.model_copy(update={'dependencies': new_deps})
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility)."""
        result = {
            "index": self.index,
            "description": self.description,
            "type": self.type.value,
            "code": self.code.model_dump(mode='json') if self.code else None,
            "conclusion": self.conclusion,
            "execution_result": self.execution_result,
            "error": self.error,
            "perception": self.perception.model_dump(mode='json') if self.perception else None,
            "status": self.status.value,
            "attempts": self.attempts,
            "was_replanned": self.was_replanned,
            "parent_index": self.parent_index,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
        return result


class PlanVersion(BaseModel):
    """Plan version with metadata and versioning support."""
    version_id: int = Field(ge=0, description="Plan version ID")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    reason: str = Field(default="", description="Reason for creating this version")
    parent_version_id: Optional[int] = Field(default=None, ge=0, description="Parent version ID if replanned")
    plan_text: List[str] = Field(default_factory=list, description="Full plan text")
    steps: List[Step] = Field(default_factory=list, description="Steps in this plan version")
    
    # Optional metrics
    estimated_duration: Optional[float] = Field(default=None, ge=0, description="Estimated duration in seconds")
    complexity_score: Optional[float] = Field(default=None, ge=0, description="Complexity score")
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    def get_step_by_index(self, index: int) -> Optional[Step]:
        """Get step by index in this plan version."""
        return next((s for s in self.steps if s.index == index), None)
    
    def get_completed_steps(self) -> List[Step]:
        """Get all completed steps in this plan version."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]
    
    def get_pending_steps(self) -> List[Step]:
        """Get all pending steps in this plan version."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]
    
    def get_failed_steps(self) -> List[Step]:
        """Get all failed steps in this plan version."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]
    
    def update_step(self, step_index: int, updated_step: Step) -> 'PlanVersion':
        """Create new plan version with updated step."""
        new_steps = []
        updated = False
        for step in self.steps:
            if step.index == step_index:
                new_steps.append(updated_step)
                updated = True
            else:
                new_steps.append(step)
        
        if not updated:
            raise ValueError(f"Step {step_index} not found in plan version {self.version_id}")
        
        return self.model_copy(update={"steps": new_steps})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility)."""
        return {
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reason": self.reason,
            "parent_version_id": self.parent_version_id,
            "plan_text": self.plan_text,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_duration": self.estimated_duration,
            "complexity_score": self.complexity_score
        }


class SessionState(BaseModel):
    """Immutable session state with proper validation."""
    original_goal_achieved: bool = Field(default=False, description="Whether goal is achieved")
    final_answer: Optional[str] = Field(default=None, description="Final answer")
    confidence: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Confidence score (0.0-1.0)"
    )
    reasoning_note: str = Field(default="", description="Reasoning note")
    solution_summary: str = Field(default="", description="Solution summary")
    
    model_config = ConfigDict(
        frozen=True,  # Immutable in Pydantic v2
        json_encoders={Decimal: str}
    )
    
    def update_with_perception(
        self,
        perception: PerceptionSnapshot,
        final_answer: Optional[str] = None
    ) -> 'SessionState':
        """Create new state from perception (immutable update)."""
        return self.model_copy(update={
            'original_goal_achieved': perception.original_goal_achieved,
            'final_answer': final_answer or perception.solution_summary,
            'confidence': perception.confidence if perception.confidence > Decimal("0.0") else self.confidence,
            'reasoning_note': perception.reasoning,
            'solution_summary': perception.solution_summary
        })


class SessionMetadata(BaseModel):
    """Session metadata for extensibility."""
    user_id: Optional[str] = Field(default=None, description="User identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    tags: List[str] = Field(default_factory=list, description="Session tags")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    
    model_config = ConfigDict(json_encoders={})


class SessionObserver(Protocol):
    """Protocol for session state change observers."""
    
    def on_state_change(
        self,
        session: 'AgentSession',
        old_state: SessionState,
        new_state: SessionState
    ) -> None:
        """Called when session state changes."""
        ...
    
    def on_plan_added(self, session: 'AgentSession', plan: PlanVersion) -> None:
        """Called when a new plan version is added."""
        ...
    
    def on_perception_added(self, session: 'AgentSession', perception: PerceptionSnapshot) -> None:
        """Called when a perception snapshot is added."""
        ...
    
    def on_lifecycle_state_change(
        self,
        session: 'AgentSession',
        old_state: str,
        new_state: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when session lifecycle state changes."""
        ...
    
    def on_step_started(self, session: 'AgentSession', step: 'Step') -> None:
        """Called when a step starts execution."""
        ...
    
    def on_step_completed(self, session: 'AgentSession', step: 'Step') -> None:
        """Called when a step completes execution."""
        ...
    
    def on_step_failed(self, session: 'AgentSession', step: 'Step', error: str) -> None:
        """Called when a step fails."""
        ...


class AgentSession:
    """Agent session managing the complete agent lifecycle with observer pattern and event logging."""
    
    def __init__(
        self,
        session_id: str,
        original_query: str,
        metadata: Optional[SessionMetadata] = None
    ):
        self.session_id = session_id
        self.original_query = original_query
        self.metadata = metadata or SessionMetadata()
        
        # Timestamps
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = self.created_at
        self.completed_at: Optional[datetime] = None
        
        # State
        self.perception: Optional[PerceptionSnapshot] = None
        self.plan_versions: List[PlanVersion] = []  # Now using PlanVersion instead of dict
        self.state: SessionState = SessionState()
        
        # State machine for lifecycle management
        from agent.session_state_machine import SessionStateMachine, SessionLifecycleState
        self._lifecycle_machine = SessionStateMachine(SessionLifecycleState.INITIALIZED)
        
        # Observers
        self._observers: List[SessionObserver] = []
        
        # Event log
        self._event_log: List[Dict[str, Any]] = []
        
        # Log initial creation
        self._log_event("session_created", {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "metadata": self.metadata.model_dump()
        })
        
        # Notify lifecycle state change
        self._notify_lifecycle_state_change(
            None,
            self._lifecycle_machine.current_state.value,
            {"session_id": self.session_id}
        )
        
        # Notify lifecycle state change
        self._notify_lifecycle_state_change(
            None,
            self._lifecycle_machine.current_state.value,
            {"session_id": self.session_id}
        )

    def add_observer(self, observer: SessionObserver) -> None:
        """Add an observer for state changes."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: SessionObserver) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_state_change(self, old_state: SessionState, new_state: SessionState) -> None:
        """Notify observers of state change."""
        for observer in self._observers:
            try:
                observer.on_state_change(self, old_state, new_state)
            except Exception as e:
                print(f"⚠️ Error notifying observer: {e}")
        
        self._log_event("state_change", {
            "old_state": old_state.model_dump(mode='json'),
            "new_state": new_state.model_dump(mode='json')
        })
    
    def _notify_lifecycle_state_change(
        self,
        old_state: Optional[str],
        new_state: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify observers of lifecycle state change."""
        for observer in self._observers:
            try:
                if hasattr(observer, 'on_lifecycle_state_change'):
                    observer.on_lifecycle_state_change(self, old_state or "unknown", new_state, data)
            except Exception as e:
                print(f"⚠️ Error notifying observer of lifecycle change: {e}")
        
        self._log_event("lifecycle_state_change", {
            "old_state": old_state,
            "new_state": new_state,
            "data": data or {}
        })
    
    def transition_lifecycle_state(
        self,
        target_state: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to a new lifecycle state.
        
        Args:
            target_state: Target state name
            data: Optional data for the transition
            
        Returns:
            True if transition succeeded, False otherwise
        """
        from agent.session_state_machine import SessionLifecycleState
        
        try:
            # Convert string to enum
            if isinstance(target_state, str):
                target_state_enum = SessionLifecycleState[target_state.upper()]
            else:
                target_state_enum = target_state
            
            old_state = self._lifecycle_machine.current_state.value
            success = self._lifecycle_machine.transition_to(target_state_enum, data)
            
            if success:
                self._notify_lifecycle_state_change(
                    old_state,
                    self._lifecycle_machine.current_state.value,
                    data
                )
            
            return success
        except (KeyError, ValueError) as e:
            print(f"⚠️ Invalid lifecycle state transition: {e}")
            return False
    
    @property
    def lifecycle_state(self) -> str:
        """Get current lifecycle state."""
        return self._lifecycle_machine.current_state.value
    
    def notify_step_started(self, step: Step) -> None:
        """Notify observers that a step has started."""
        for observer in self._observers:
            try:
                if hasattr(observer, 'on_step_started'):
                    observer.on_step_started(self, step)
            except Exception as e:
                print(f"⚠️ Error notifying observer of step start: {e}")
        
        self._log_event("step_started", {
            "step_index": step.index,
            "step_type": step.type.value,
            "description": step.description
        })
    
    def notify_step_completed(self, step: Step) -> None:
        """Notify observers that a step has completed."""
        for observer in self._observers:
            try:
                if hasattr(observer, 'on_step_completed'):
                    observer.on_step_completed(self, step)
            except Exception as e:
                print(f"⚠️ Error notifying observer of step completion: {e}")
        
        self._log_event("step_completed", {
            "step_index": step.index,
            "step_type": step.type.value,
            "status": step.status.value
        })
    
    def notify_step_failed(self, step: Step, error: str) -> None:
        """Notify observers that a step has failed."""
        for observer in self._observers:
            try:
                if hasattr(observer, 'on_step_failed'):
                    observer.on_step_failed(self, step, error)
            except Exception as e:
                print(f"⚠️ Error notifying observer of step failure: {e}")
        
        self._log_event("step_failed", {
            "step_index": step.index,
            "step_type": step.type.value,
            "error": error
        })

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to the audit trail."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self._event_log.append(event)
        self.updated_at = datetime.now()
        
        # Limit event log size to prevent memory issues (keep last 1000 events)
        if len(self._event_log) > 1000:
            self._event_log = self._event_log[-1000:]

    def get_event_log(self, event_type: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event log entries, optionally filtered by type and limited in count."""
        events = self._event_log
        if event_type:
            events = [e for e in events if e.get("type") == event_type]
        if limit:
            events = events[-limit:]  # Get last N events
        return events

    def add_perception(self, snapshot: PerceptionSnapshot) -> None:
        """Add a perception snapshot with validation."""
        if not isinstance(snapshot, PerceptionSnapshot):
            raise TypeError(f"snapshot must be PerceptionSnapshot, got {type(snapshot)}")
        
        self.perception = snapshot
        self._log_event("perception_added", {
            "snapshot": snapshot.model_dump(mode='json')
        })
        
        # Transition lifecycle state
        self.transition_lifecycle_state("PERCEPTION_RECEIVED", {
            "perception_id": id(snapshot),
            "original_goal_achieved": snapshot.original_goal_achieved
        })
        
        # Notify observers
        for observer in self._observers:
            try:
                observer.on_perception_added(self, snapshot)
            except Exception as e:
                print(f"⚠️ Error notifying observer: {e}")

    def add_plan_version(
        self,
        plan_texts: List[str],
        steps: List[Step],
        reason: str = ""
    ) -> Optional[Step]:
        """Add a new plan version with metadata."""
        version_id = len(self.plan_versions)
        parent_version_id = version_id - 1 if version_id > 0 else None
        
        plan = PlanVersion(
            version_id=version_id,
            plan_text=plan_texts,
            steps=[step.model_copy() for step in steps],  # Deep copy using Pydantic
            reason=reason,
            parent_version_id=parent_version_id
        )
        self.plan_versions.append(plan)
        
        self._log_event("plan_added", {
            "version_id": version_id,
            "reason": reason,
            "plan_text_count": len(plan_texts),
            "steps_count": len(steps)
        })
        
        # Transition to PLANNING state if not already there
        # Can transition from PERCEPTION_RECEIVED or EXECUTING (replanning)
        current_state = self._lifecycle_machine.current_state.value
        if current_state == "perception_received" or current_state == "executing":
            self.transition_lifecycle_state("PLANNING", {
                "plan_version": version_id,
                "reason": reason
            })
        
        # Notify observers
        for observer in self._observers:
            try:
                observer.on_plan_added(self, plan)
            except Exception as e:
                print(f"⚠️ Error notifying observer: {e}")
        
        return steps[0] if steps else None
    
    def update_step(self, step_index: int, updated_step: Step) -> bool:
        """Update a step in the current plan version."""
        if not self.plan_versions:
            return False
        
        current_plan = self.plan_versions[-1]
        try:
            new_plan = current_plan.update_step(step_index, updated_step)
            self.plan_versions[-1] = new_plan
            
            self._log_event("step_updated", {
                "step_index": step_index,
                "status": updated_step.status.value,
                "plan_version_id": current_plan.version_id
            })
            return True
        except ValueError:
            return False

    def get_current_plan(self) -> Optional[PlanVersion]:
        """Get the current (latest) plan version."""
        return self.plan_versions[-1] if self.plan_versions else None

    def get_next_step_index(self) -> int:
        """Get the next step index across all plan versions."""
        return sum(len(v.steps) for v in self.plan_versions)

    def get_all_steps(self) -> List[Step]:
        """Get all steps from all plan versions."""
        steps = []
        for plan in self.plan_versions:
            steps.extend(plan.steps)
        return steps

    def get_step_by_index(self, index: int) -> Optional[Step]:
        """Get step by index across all plan versions."""
        for plan in self.plan_versions:
            step = plan.get_step_by_index(index)
            if step:
                return step
        return None

    def to_json(self) -> Dict[str, Any]:
        """Convert session to JSON-serializable dictionary."""
        result = {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "metadata": self.metadata.model_dump(mode='json'),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "perception": self.perception.model_dump(mode='json') if self.perception else None,
            "plan_versions": [p.to_dict() for p in self.plan_versions],
            "state": self.state.model_dump(mode='json'),
            "state_snapshot": self.get_snapshot_summary(),
            "event_log": self._event_log[-100:]  # Include last 100 events
        }
        # Ensure all Decimal values are converted to strings
        return _convert_decimals_to_str(result)

    def get_snapshot_summary(self) -> Dict[str, Any]:
        """Get a summary snapshot of the session."""
        return {
            "session_id": self.session_id,
            "query": self.original_query,
            "final_plan": self.plan_versions[-1].plan_text if self.plan_versions else [],
            "final_steps": [
                s.to_dict()
                for version in self.plan_versions
                for s in version.steps
                if s.status == StepStatus.COMPLETED
            ],
            "final_answer": self.state.final_answer,
            "confidence": str(self.state.confidence),  # Convert Decimal to string
            "reasoning_note": self.state.reasoning_note,
            "original_goal_achieved": self.state.original_goal_achieved,
            "solution_summary": self.state.solution_summary,
            "duration_seconds": (self.completed_at - self.created_at).total_seconds() if self.completed_at else None,
            "total_plan_versions": len(self.plan_versions),
            "total_events": len(self._event_log)
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with metrics."""
        all_steps = self.get_all_steps()
        return {
            "total_steps": len(all_steps),
            "completed_steps": len([s for s in all_steps if s.status == StepStatus.COMPLETED]),
            "failed_steps": len([s for s in all_steps if s.status == StepStatus.FAILED]),
            "pending_steps": len([s for s in all_steps if s.status == StepStatus.PENDING]),
            "plan_versions": len(self.plan_versions),
            "replan_count": len([s for s in all_steps if s.was_replanned]),
            "total_attempts": sum(s.attempts for s in all_steps),
            "duration_seconds": (self.completed_at - self.created_at).total_seconds() if self.completed_at else None
        }

    def mark_complete(
        self,
        perception: PerceptionSnapshot,
        final_answer: Optional[str] = None,
        fallback_confidence: Optional[Decimal] = None
    ) -> None:
        """Mark session as complete with perception and final answer (immutable state update)."""
        # Validation
        if not isinstance(perception, PerceptionSnapshot):
            raise TypeError(f"perception must be PerceptionSnapshot, got {type(perception)}")
        
        old_state = self.state
        self.state = self.state.update_with_perception(perception, final_answer)
        self.completed_at = datetime.now()
        
        # Transition to completed state (from EXECUTING or PLANNING)
        current_state = self._lifecycle_machine.current_state.value
        if current_state in ("executing", "planning"):
            self.transition_lifecycle_state("COMPLETED", {
                "final_answer": final_answer or perception.solution_summary,
                "confidence": str(self.state.confidence),
                "original_goal_achieved": self.state.original_goal_achieved,
                "duration_seconds": (self.completed_at - self.created_at).total_seconds() if self.completed_at else None
            })
        # If already in COMPLETED or other terminal state, skip transition
        
        self._log_event("session_completed", {
            "final_answer": final_answer or perception.solution_summary,
            "confidence": str(self.state.confidence),
            "original_goal_achieved": self.state.original_goal_achieved,
            "duration_seconds": (self.completed_at - self.created_at).total_seconds() if self.completed_at else None
        })
        
        # Notify observers of state change
        self._notify_state_change(old_state, self.state)



    def simulate_live(self, delay: float = 1.2) -> None:
        """Simulate live session trace with delays."""
        print("\n=== LIVE AGENT SESSION TRACE ===")
        print(f"Session ID: {self.session_id}")
        print(f"Query: {self.original_query}")
        time.sleep(delay)

        if self.perception:
            print("\n[Perception 0] Initial ERORLL:")
            print(f"  {self.perception.model_dump()}")
            time.sleep(delay)

        for i, version in enumerate(self.plan_versions):
            print(f"\n[Decision Plan Text: V{version.version_id + 1}]:")
            if version.reason:
                print(f"  Reason: {version.reason}")
            for j, p in enumerate(version.plan_text):
                print(f"  Step {j}: {p}")
            time.sleep(delay)

            for step in version.steps:
                print(f"\n[Step {step.index}] {step.description}")
                time.sleep(delay / 1.5)

                print(f"  Type: {step.type.value}")
                if step.code:
                    print(f"  Tool → {step.code.tool_name} | Args → {step.code.tool_arguments}")
                if step.execution_result:
                    print(f"  Execution Result: {step.execution_result}")
                if step.conclusion:
                    print(f"  Conclusion: {step.conclusion}")
                if step.error:
                    print(f"  Error: {step.error}")
                if step.perception:
                    print("  Perception ERORLL:")
                    for k, v in step.perception.model_dump().items():
                        print(f"    {k}: {v}")
                print(f"  Status: {step.status.value}")
                if step.was_replanned:
                    print(f"  (Replanned from Step {step.parent_index})")
                if step.dependencies:
                    print(f"  Dependencies: {step.dependencies}")
                if step.attempts > 1:
                    print(f"  Attempts: {step.attempts}")
                if step.started_at:
                    print(f"  Started: {step.started_at.isoformat()}")
                if step.completed_at:
                    print(f"  Completed: {step.completed_at.isoformat()}")
                time.sleep(delay)

        print("\n[Session Snapshot]:")
        print(json.dumps(self.get_snapshot_summary(), indent=2))
