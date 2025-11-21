"""
SessionStateMachine - State machine for managing agent session lifecycle.

This module provides a state machine for tracking and managing the lifecycle
of an agent session, ensuring valid state transitions and proper validation.
"""

from enum import Enum
from typing import Optional, Set, Dict, Any, Callable, List
from datetime import datetime


class SessionLifecycleState(str, Enum):
    """Valid states in the session lifecycle."""
    INITIALIZED = "initialized"
    PERCEPTION_RECEIVED = "perception_received"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionStateMachine:
    """
    State machine for managing agent session lifecycle.
    
    Ensures valid state transitions and provides hooks for state change handlers.
    
    Example:
        machine = SessionStateMachine()
        machine.transition_to(SessionLifecycleState.PERCEPTION_RECEIVED)
        machine.add_transition_handler(SessionLifecycleState.COMPLETED, on_complete)
    """
    
    # Valid state transitions: {from_state: {to_states}}
    VALID_TRANSITIONS: Dict[SessionLifecycleState, Set[SessionLifecycleState]] = {
        SessionLifecycleState.INITIALIZED: {
            SessionLifecycleState.PERCEPTION_RECEIVED,
            SessionLifecycleState.FAILED,
            SessionLifecycleState.CANCELLED
        },
        SessionLifecycleState.PERCEPTION_RECEIVED: {
            SessionLifecycleState.PLANNING,
            SessionLifecycleState.FAILED,
            SessionLifecycleState.CANCELLED
        },
        SessionLifecycleState.PLANNING: {
            SessionLifecycleState.EXECUTING,
            SessionLifecycleState.FAILED,
            SessionLifecycleState.CANCELLED
        },
        SessionLifecycleState.EXECUTING: {
            SessionLifecycleState.PLANNING,  # Replanning
            SessionLifecycleState.PAUSED,
            SessionLifecycleState.COMPLETED,
            SessionLifecycleState.FAILED,
            SessionLifecycleState.CANCELLED
        },
        SessionLifecycleState.PAUSED: {
            SessionLifecycleState.EXECUTING,
            SessionLifecycleState.CANCELLED
        },
        # Terminal states - no transitions allowed
        SessionLifecycleState.COMPLETED: set(),
        SessionLifecycleState.FAILED: set(),
        SessionLifecycleState.CANCELLED: set(),
    }
    
    def __init__(self, initial_state: SessionLifecycleState = SessionLifecycleState.INITIALIZED):
        """
        Initialize the state machine.
        
        Args:
            initial_state: Initial state (default: INITIALIZED)
        """
        self._current_state = initial_state
        self._previous_state: Optional[SessionLifecycleState] = None
        self._state_history: List[tuple[SessionLifecycleState, datetime]] = [
            (initial_state, datetime.now())
        ]
        self._transition_handlers: Dict[SessionLifecycleState, List[Callable]] = {}
        self._transition_data: Dict[SessionLifecycleState, Dict[str, Any]] = {}
    
    @property
    def current_state(self) -> SessionLifecycleState:
        """Get the current state."""
        return self._current_state
    
    @property
    def previous_state(self) -> Optional[SessionLifecycleState]:
        """Get the previous state."""
        return self._previous_state
    
    @property
    def state_history(self) -> List[tuple[SessionLifecycleState, datetime]]:
        """Get the state transition history."""
        return self._state_history.copy()
    
    def can_transition_to(self, target_state: SessionLifecycleState) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target_state: Target state to check
            
        Returns:
            True if transition is valid, False otherwise
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._current_state, set())
        return target_state in valid_targets
    
    def transition_to(
        self,
        target_state: SessionLifecycleState,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to transition to target state.
        
        Args:
            target_state: Target state
            data: Optional data to associate with this transition
            
        Returns:
            True if transition succeeded, False otherwise
            
        Raises:
            ValueError: If transition is invalid
        """
        if not self.can_transition_to(target_state):
            raise ValueError(
                f"Invalid state transition: {self._current_state.value} -> {target_state.value}. "
                f"Valid transitions from {self._current_state.value}: "
                f"{[s.value for s in self.VALID_TRANSITIONS.get(self._current_state, set())]}"
            )
        
        # Store previous state
        self._previous_state = self._current_state
        
        # Update state
        self._current_state = target_state
        self._state_history.append((target_state, datetime.now()))
        
        # Store transition data
        if data:
            self._transition_data[target_state] = data
        
        # Call transition handlers
        handlers = self._transition_handlers.get(target_state, [])
        for handler in handlers:
            try:
                handler(self._previous_state, target_state, data)
            except Exception as e:
                # Log but don't fail transition
                print(f"⚠️ Error in state transition handler: {e}")
        
        return True
    
    def add_transition_handler(
        self,
        target_state: SessionLifecycleState,
        handler: Callable[[Optional[SessionLifecycleState], SessionLifecycleState, Optional[Dict[str, Any]]], None]
    ) -> None:
        """
        Add a handler to be called when transitioning to a specific state.
        
        Args:
            target_state: State to handle
            handler: Function called as handler(previous_state, new_state, data)
        """
        if target_state not in self._transition_handlers:
            self._transition_handlers[target_state] = []
        self._transition_handlers[target_state].append(handler)
    
    def remove_transition_handler(
        self,
        target_state: SessionLifecycleState,
        handler: Callable
    ) -> None:
        """Remove a transition handler."""
        if target_state in self._transition_handlers:
            try:
                self._transition_handlers[target_state].remove(handler)
            except ValueError:
                pass  # Handler not found
    
    def get_transition_data(self, state: SessionLifecycleState) -> Optional[Dict[str, Any]]:
        """Get data associated with a state transition."""
        return self._transition_data.get(state)
    
    def is_terminal_state(self) -> bool:
        """Check if current state is a terminal state."""
        terminal_states = {
            SessionLifecycleState.COMPLETED,
            SessionLifecycleState.FAILED,
            SessionLifecycleState.CANCELLED
        }
        return self._current_state in terminal_states
    
    def reset(self, new_initial_state: SessionLifecycleState = SessionLifecycleState.INITIALIZED) -> None:
        """Reset the state machine to initial state."""
        self._previous_state = None
        self._current_state = new_initial_state
        self._state_history = [(new_initial_state, datetime.now())]
        self._transition_data.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SessionStateMachine(state={self._current_state.value}, history_length={len(self._state_history)})"

