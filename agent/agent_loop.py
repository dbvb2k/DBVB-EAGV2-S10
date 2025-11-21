import uuid
import json
import ast
import re
from typing import Optional, List
from datetime import datetime
from perception.perception import Perception
from decision.decision import Decision
from action.executor import run_user_code
from agent.agentSession import AgentSession, PerceptionSnapshot, Step, ToolCode, StepType, StepStatus
from memory.session_log import live_update_session as live_update_session_async
from memory.memory_search import MemorySearch
from mcp_servers.multiMCP import MultiMCP


# Configuration constants
from config.settings import get_config

_config_manager = get_config()
_config = _config_manager.config
MAX_PREVIOUS_FAILURE_STEPS = _config.memory.max_previous_failure_steps
MAX_ITERATIONS = _config.agent.max_iterations

NUM_DASHES = 150 

class AgentLoop:
    """
    Main agent loop orchestrating perception, decision, and action execution.
    
    This class manages the complete agent lifecycle from query to final answer,
    handling memory search, perception, planning, execution, and evaluation.
    """
    
    def __init__(
        self, 
        perception_prompt_path: str, 
        decision_prompt_path: str, 
        multi_mcp: MultiMCP, 
        strategy: str = "exploratory"
    ):
        self.perception = Perception(perception_prompt_path)
        self.decision = Decision(decision_prompt_path, multi_mcp)
        self.multi_mcp = multi_mcp
        self.strategy = strategy

    async def run(self, query: str) -> AgentSession:
        """
        Execute the main agent loop for a given query.
        
        Args:
            query: User query string
            
        Returns:
            AgentSession with complete execution trace
        """
        session = AgentSession(session_id=str(uuid.uuid4()), original_query=query)
        session_memory = []  # Track failures within this session
        iteration_count = 0
        
        self.log_session_start(session, query)

        # Step 0: Memory Search
        memory_results = await self.search_memory(query)
        
        # Step 1: Initial Perception
        perception_result = self.run_perception(
            query=query,
            memory_results=memory_results,
            session_memory=session_memory,
            snapshot_type="user_query"
        )
        session.add_perception(PerceptionSnapshot(**perception_result))

        # Early exit if perception is confident
        if perception_result.get("original_goal_achieved"):
            await self.handle_perception_completion(session, perception_result)
            return session

        # Step 2: Initial Decision/Planning
        decision_output = self.make_initial_decision(query, perception_result)
        step = session.add_plan_version(
            decision_output["plan_text"], 
            [self.create_step(decision_output)]
        )
        await live_update_session_async(session)
        self.print_plan_version(session, len(session.plan_versions))

        # Step 3: Execution Loop
        while step and iteration_count < MAX_ITERATIONS:
            iteration_count += 1
            step_result = await self.execute_step(step, session, session_memory)
            
            if step_result is None:
                break  # CONCLUDE or NOP cases
            
            step = await self.evaluate_step(step_result, session, query, session_memory)
        
        if iteration_count >= MAX_ITERATIONS:
            print(f"\n⚠️ Maximum iterations ({MAX_ITERATIONS}) reached. Stopping execution.")
            from decimal import Decimal
            old_state = session.state
            session.state = session.state.model_copy(update={
                "original_goal_achieved": False,
                "final_answer": "Execution stopped due to maximum iteration limit.",
                "confidence": Decimal("0.0"),
                "reasoning_note": "Agent loop exceeded maximum iterations.",
                "solution_summary": "Unable to complete task within iteration limits."
            })
            session.completed_at = datetime.now()
            session._notify_state_change(old_state, session.state)
            session._log_event("session_failed_max_iterations", {
                "max_iterations": MAX_ITERATIONS,
                "iteration_count": iteration_count
            })
            await live_update_session_async(session)

        return session

    def log_session_start(self, session: AgentSession, query: str) -> None:
        """Log session initialization."""
        print("\n=== LIVE AGENT SESSION TRACE ===")
        print(f"Session ID: {session.session_id}")
        print(f"Query: {query}")

    async def search_memory(self, query: str) -> list:
        """
        Search historical memory for relevant past sessions.
        
        Args:
            query: Search query
            
        Returns:
            List of matching memory entries
        """
        print("Searching Recent Conversation History")
        searcher = MemorySearch()
        results = await searcher.search_memory(query)
        
        if not results:
            print("❌ No matching memory entries found.\n")
        else:
            print("\n🎯 Top Matches:\n")
            for i, res in enumerate(results, 1):
                print(f"[{i}] File: {res['file']}")
                print(f"    Query: {res['query']}")
                print(f"    Result Requirement: {res['result_requirement']}")
                print(f"    Summary: {res['solution_summary']}\n")
        
        return results

    def _extract_tool_names_from_code(self, code: str) -> List[str]:
        """Extract tool/function names called in the code."""
        tool_names = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        tool_names.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # Handle method calls like obj.method()
                        tool_names.append(node.func.attr)
        except SyntaxError:
            # If parsing fails, try regex fallback
            # Look for function call patterns
            pattern = r'(\w+)\s*\('
            matches = re.findall(pattern, code)
            tool_names.extend(matches)
        return list(set(tool_names))  # Remove duplicates
    
    def _get_tool_use_summary(self, step: Step, executor_response: dict) -> str:
        """Generate tool use summary for the current step."""
        if step.type != StepType.CODE or not step.code:
            return "No tool used - direct conclusion or no-op step"
        
        code = step.code.tool_arguments.get("code", "")
        tool_names = self._extract_tool_names_from_code(code)
        
        # Check if execution was successful
        status = executor_response.get("status", "unknown")
        
        if status == "success":
            if tool_names:
                # Filter out built-in functions and common Python functions
                builtins = {"print", "len", "str", "int", "float", "list", "dict", "range", "sum", "return"}
                actual_tools = [t for t in tool_names if t not in builtins and not t.startswith("_")]
                if actual_tools:
                    return f"{', '.join(actual_tools)} - Success"
                else:
                    return "No tool used - direct computation/string manipulation"
            else:
                return "No tool used - direct computation/string manipulation"
        else:
            # Execution failed
            error = executor_response.get("error", "Unknown error")
            error_type = executor_response.get("error_type", "error")
            if tool_names:
                builtins = {"print", "len", "str", "int", "float", "list", "dict", "range", "sum", "return"}
                actual_tools = [t for t in tool_names if t not in builtins and not t.startswith("_")]
                if actual_tools:
                    return f"{', '.join(actual_tools)} - Failed due to {error_type}: {error[:100]}"
            return f"Execution failed: {error_type}: {error[:100]}"
    
    def run_perception(
        self, 
        query: str, 
        memory_results: list, 
        session_memory: Optional[list] = None,
        snapshot_type: str = "user_query",
        current_plan: Optional[list] = None,
        current_step_tool_summary: Optional[str] = None
    ) -> dict:
        """
        Run perception on given input.
        
        Args:
            query: Input to process
            memory_results: Historical memory results
            session_memory: Current session memory
            snapshot_type: Type of perception snapshot
            current_plan: Current execution plan
            
        Returns:
            Perception result dictionary
        """
        combined_memory = (memory_results or []) + (session_memory or [])
        perception_input = self.perception.build_perception_input(
            raw_input=query,
            memory=combined_memory,
            current_plan=current_plan or "",
            snapshot_type=snapshot_type,
            current_step_tool_summary=current_step_tool_summary
        )
        perception_result = self.perception.run(perception_input)
        
        print(f"\n[Perception Result ({snapshot_type})]:")
        print(json.dumps(perception_result, indent=2, ensure_ascii=False))
        
        return perception_result

    async def handle_perception_completion(self, session: AgentSession, perception_result: dict) -> None:
        """Handle early completion when perception is confident."""
        print("\n✅ Perception has already fully answered the query.")
        from decimal import Decimal
        
        # Create PerceptionSnapshot from result
        perception = PerceptionSnapshot(**perception_result)
        
        # Update state using immutable update
        old_state = session.state
        session.state = session.state.model_copy(update={
            "original_goal_achieved": True,
            "final_answer": perception_result.get("solution_summary", "Answer ready."),
            "confidence": Decimal(str(perception_result.get("confidence", "0.95"))),
            "reasoning_note": perception_result.get("reasoning", "Fully handled by initial perception."),
            "solution_summary": perception_result.get("solution_summary", "Answer ready.")
        })
        session.completed_at = datetime.now()
        session._notify_state_change(old_state, session.state)
        session._log_event("session_completed_early", {
            "reason": "Perception completed query directly",
            "confidence": str(session.state.confidence)
        })
        await live_update_session_async(session)

    def make_initial_decision(self, query: str, perception_result: dict) -> dict:
        """Generate initial decision/plan based on perception."""
        decision_input = {
            "plan_mode": "initial",
            "planning_strategy": self.strategy,
            "original_query": query,
            "perception": perception_result
        }
        return self.decision.run(decision_input)

    def create_step(self, decision_output: dict) -> Step:
        """Create a Step object from decision output using StepBuilder."""
        from agent.step_builder import StepBuilder
        
        builder = StepBuilder()
        builder.from_decision_output(decision_output)
        return builder.build()

    async def execute_step(
        self, 
        step: Step, 
        session: AgentSession, 
        session_memory: list
    ) -> Optional[Step]:
        """
        Execute a single step.
        
        Returns:
            Step object if execution continues, None if should stop
        """
        print(f"\n[Step {step.index}] {step.description}")
        
        # Notify step started
        session.notify_step_started(step)
        
        # Transition to executing state (from PLANNING or EXECUTING if already executing)
        current_state = session.lifecycle_state
        if current_state == "planning" or current_state == "executing":
            session.transition_lifecycle_state("EXECUTING", {
                "step_index": step.index,
                "step_type": step.type.value
            })
        # If already in EXECUTING, we're good (replanning case)

        try:
            if step.type == StepType.CODE:
                result = await self._execute_code_step(step, session, session_memory)
            elif step.type == StepType.CONCLUDE:
                result = await self._execute_conclude_step(step, session, session_memory)
            elif step.type in (StepType.NOP, StepType.NOOP):
                result = await self._execute_nop_step(step, session)
            else:
                print(f"⚠️ Unknown step type: {step.type}")
                result = step.mark_failed(f"Unknown step type: {step.type}")
            
            # Notify step completion or failure
            if result and result.status == StepStatus.COMPLETED:
                session.notify_step_completed(result)
            elif result and result.status == StepStatus.FAILED:
                session.notify_step_failed(result, result.error or "Unknown error")
            
            return result
        except Exception as e:
            # Notify step failure
            failed_step = step.mark_failed(str(e))
            session.notify_step_failed(failed_step, str(e))
            session.transition_lifecycle_state("FAILED", {
                "step_index": step.index,
                "error": str(e)
            })
            raise

    async def _execute_code_step(
        self, 
        step: Step, 
        session: AgentSession, 
        session_memory: list
    ) -> Step:
        """Execute a CODE step."""
        print("-" * NUM_DASHES)
        print("[EXECUTING CODE]")
        print(step.code.tool_arguments["code"])
        
        try:
            executor_response = await run_user_code(
                step.code.tool_arguments["code"], 
                self.multi_mcp
            )
            step = step.mark_completed(executor_response)

            # Run perception on execution result
            current_plan_text = session.plan_versions[-1].plan_text if session.plan_versions else []
            tool_summary = self._get_tool_use_summary(step, executor_response)
            perception_result = self.run_perception(
                query=executor_response.get('result', 'Tool Failed'),
                memory_results=session_memory,
                current_plan=current_plan_text,
                snapshot_type="step_result",
                current_step_tool_summary=tool_summary
            )
            step = step.model_copy(update={'perception': PerceptionSnapshot(**perception_result)})

            # Track failures in session memory
            if not step.perception or not step.perception.local_goal_achieved:
                failure_memory = {
                    "query": step.description,
                    "result_requirement": "Tool failed",
                    "solution_summary": str(step.execution_result)[:300]
                }
                session_memory.append(failure_memory)
                
                # Limit session memory size
                if len(session_memory) > MAX_PREVIOUS_FAILURE_STEPS:
                    session_memory.pop(0)

            await live_update_session_async(session)
            return step
            
        except Exception as e:
            print(f"❌ Error executing code step: {e}")
            step = step.mark_failed(str(e))
            step = step.model_copy(update={
                'execution_result': {"status": "error", "error": str(e)}
            })
            await live_update_session_async(session)
            return step

    async def _execute_conclude_step(
        self, 
        step: Step, 
        session: AgentSession, 
        session_memory: list
    ) -> None:
        """Execute a CONCLUDE step."""
        print(f"\n💡 Conclusion: {step.conclusion}")

        # For CONCLUDE steps, no tool was used
        tool_summary = "No tool used - conclusion step"
        perception_result = self.run_perception(
            query=step.conclusion,
            memory_results=session_memory,
            current_plan=session.plan_versions[-1].plan_text if session.plan_versions else [],
            snapshot_type="step_result",
            current_step_tool_summary=tool_summary
        )
        perception = PerceptionSnapshot(**perception_result)
        
        # Handle "Not ready yet" case
        if 'Not ready yet' in perception_result.get('solution_summary', ''):
            perception_result['solution_summary'] = (
                perception_result.get('reasoning', '') + "\n" +
                perception_result.get('local_reasoning', '') +
                "\nIf you disagree, try to be more specific in your query.\n"
            )
            perception = PerceptionSnapshot(**perception_result)
        
        step = step.mark_completed(step.conclusion, perception)
        session.add_perception(perception)
        session.mark_complete(perception, final_answer=step.conclusion)
        await live_update_session_async(session)
        return None  # Signal to stop execution

    async def _execute_nop_step(self, step: Step, session: AgentSession) -> None:
        """Execute a NOP (no operation) step."""
        print(f"\n❓ Clarification needed: {step.description}")
        step = step.model_copy(update={'status': StepStatus.CLARIFICATION_NEEDED})
        await live_update_session_async(session)
        return None  # Signal to stop execution

    async def evaluate_step(
        self, 
        step: Step, 
        session: AgentSession, 
        query: str,
        session_memory: list
    ) -> Optional[Step]:
        """
        Evaluate step result and determine next action.
        
        Returns:
            Next Step to execute, or None if should stop
        """
        if not step.perception:
            print("⚠️ Step has no perception result. Replanning...")
            return self._replan(session, query, step)

        if step.perception.original_goal_achieved:
            print("\n✅ Original goal achieved.")
            session.mark_complete(step.perception)
            await live_update_session_async(session)
            return None
        
        elif step.perception.local_goal_achieved:
            # Check if we have a good solution summary that might indicate goal achievement
            solution_summary = step.perception.solution_summary or ""
            has_good_summary = (
                solution_summary 
                and solution_summary.lower() not in ("not ready yet", "no summary", "")
                and len(solution_summary.strip()) > 20  # Has substantial content
            )
            
            # Proceed to next step
            next_step = self.get_next_step(session, query, step)
            
            # If no more steps in plan but goal not achieved
            if next_step is None:
                # Check if goal is achieved
                if step.perception.original_goal_achieved:
                    # Goal achieved but somehow no more steps - mark as complete
                    print("\n✅ Goal achieved with current steps.")
                    session.mark_complete(step.perception)
                    await live_update_session_async(session)
                    return None
                else:
                    # Goal not achieved and no more steps
                    # If we have a good summary, create a CONCLUDE step instead of extending
                    if has_good_summary:
                        print(f"\n💡 Good summary available. Creating CONCLUDE step...")
                        print(f"   Summary: {solution_summary[:100]}...")
                        conclude_step = self._create_conclude_step(session, query, step, solution_summary)
                        if conclude_step:
                            print(f"   ✅ CONCLUDE step created.")
                            return conclude_step
                    
                    # Otherwise, extend the plan
                    print(f"\n📋 No more steps in current plan, but goal not yet achieved.")
                    print(f"   Current goal status: original_goal_achieved={step.perception.original_goal_achieved}")
                    print(f"   Extending plan with additional steps...")
                    extended_step = self._replan(session, query, step, reason="Extending plan - goal not yet achieved")
                    if extended_step:
                        # Validate the extended step before using it
                        if extended_step.description and extended_step.description != "Missing from LLM response":
                            print(f"   ✅ Plan extended. New step {extended_step.index} created.")
                            return extended_step
                        else:
                            print(f"   ⚠️ Invalid step from decision module: '{extended_step.description}'")
                            # Try to conclude with summary if available
                            if has_good_summary:
                                print(f"   Attempting to conclude with available summary...")
                                return self._create_conclude_step(session, query, step, solution_summary)
                            else:
                                print(f"   ❌ Failed to extend plan and no summary available.")
                                return None
                    else:
                        print(f"   ⚠️ Failed to extend plan.")
                        # Try to conclude with summary if available
                        if has_good_summary:
                            return self._create_conclude_step(session, query, step, solution_summary)
                        return None
            
            return next_step
        else:
            # Step failed or unhelpful - replan
            print(f"\n🔁 Step {step.index} failed or unhelpful. Replanning...")
            return self._replan(session, query, step)

    def get_next_step(self, session: AgentSession, query: str, step: Step) -> Optional[Step]:
        """Get the next step in the current plan."""
        next_index = step.index + 1
        if not session.plan_versions:
            return None
        current_plan = session.plan_versions[-1]
        total_steps = len(current_plan.plan_text)
        
        if next_index < total_steps:
            print(f"\n➡️ Proceeding to Step {next_index}...")
            
            # Include perception context for better decision making
            perception_context = None
            if step.perception:
                perception_context = {
                    "original_goal_achieved": step.perception.original_goal_achieved,
                    "local_goal_achieved": step.perception.local_goal_achieved,
                    "reasoning": step.perception.reasoning,
                    "local_reasoning": step.perception.local_reasoning,
                    "solution_summary": step.perception.solution_summary,
                    "result_requirement": step.perception.result_requirement
                }
            
            # Request next step from existing plan (don't extend yet)
            decision_output = self.decision.run({
                "plan_mode": "mid_session",
                "planning_strategy": self.strategy,
                "original_query": query,
                "current_plan_version": len(session.plan_versions),
                "current_plan": current_plan.plan_text,
                "completed_steps": [
                    s.to_dict() 
                    for s in current_plan.steps 
                    if s.status == StepStatus.COMPLETED
                ],
                "current_step": step.to_dict(),
                "perception": perception_context,
                "extend_plan": False  # Just getting next step, not extending
            })
            
            next_step = session.add_plan_version(
                decision_output["plan_text"],
                [self.create_step(decision_output)],
                reason=f"Continuing to step {next_index} of existing plan"
            )
            self.print_plan_version(session, len(session.plan_versions))
            return next_step
        else:
            print("\n✅ No more steps in current plan.")
            return None

    def _create_conclude_step(
        self,
        session: AgentSession,
        query: str,
        step: Step,
        summary: str
    ) -> Optional[Step]:
        """Create a CONCLUDE step with the provided summary."""
        from agent.step_builder import StepBuilder
        from agent.agentSession import StepType
        
        # Determine the next step index
        next_index = step.index + 1
        if session.plan_versions:
            current_plan = session.plan_versions[-1]
            # Use the next index after all current plan steps
            next_index = len(current_plan.plan_text)
        
        # Create CONCLUDE step with summary
        conclude_step = (StepBuilder()
            .with_index(next_index)
            .with_description("Provide final answer based on the summary")
            .with_type(StepType.CONCLUDE)
            .with_conclusion(summary)
            .build())
        
        # Add plan version with CONCLUDE step
        plan_texts = []
        if session.plan_versions:
            plan_texts = session.plan_versions[-1].plan_text.copy()
        plan_texts.append(f"Step {next_index}: Provide final answer based on summary.")
        
        new_step = session.add_plan_version(
            plan_texts,
            [conclude_step],
            reason="Concluding with available summary"
        )
        self.print_plan_version(session, len(session.plan_versions))
        return new_step
    
    def _replan(
        self, 
        session: AgentSession, 
        query: str, 
        step: Step,
        reason: Optional[str] = None
    ) -> Optional[Step]:
        """Generate a new plan after step failure or when extending plan."""
        if not session.plan_versions:
            return None
        current_plan = session.plan_versions[-1]
        
        # Include perception result to help decision module understand context
        perception_context = None
        if step.perception:
            perception_context = {
                "original_goal_achieved": step.perception.original_goal_achieved,
                "local_goal_achieved": step.perception.local_goal_achieved,
                "reasoning": step.perception.reasoning,
                "local_reasoning": step.perception.local_reasoning,
                "solution_summary": step.perception.solution_summary,
                "result_requirement": step.perception.result_requirement
            }
        
        decision_output = self.decision.run({
            "plan_mode": "mid_session",
            "planning_strategy": self.strategy,
            "original_query": query,
            "current_plan_version": len(session.plan_versions),
            "current_plan": current_plan.plan_text,
            "completed_steps": [
                s.to_dict() 
                for s in current_plan.steps 
                if s.status == StepStatus.COMPLETED
            ],
            "current_step": step.to_dict(),
            "perception": perception_context,
            "extend_plan": True,  # Signal that we want to extend the plan
            "reason": reason or "Replanning"
        })
        
        # Create step and validate it
        try:
            created_step = self.create_step(decision_output)
            
            # Validate the created step
            if not created_step.description or created_step.description == "Missing from LLM response":
                print(f"⚠️ Invalid step description from decision module: '{created_step.description}'")
                # Try to create a default valid step based on the last step's result
                if step.perception and step.perception.solution_summary:
                    solution_summary = step.perception.solution_summary
                    if solution_summary and solution_summary.lower() not in ("not ready yet", "no summary", ""):
                        print(f"   Attempting to create CONCLUDE step with available summary...")
                        return self._create_conclude_step(session, query, step, solution_summary)
                
                # If no summary, return None to signal failure
                print(f"   ❌ Cannot create valid step from decision output.")
                return None
            
            new_step = session.add_plan_version(
                decision_output["plan_text"],
                [created_step],
                reason=reason or f"Replanned after step {step.index} failure"
            )
            self.print_plan_version(session, len(session.plan_versions))
            return new_step
        except Exception as e:
            print(f"❌ Error creating step from decision output: {e}")
            # Try to conclude with summary if available
            if step.perception and step.perception.solution_summary:
                solution_summary = step.perception.solution_summary
                if solution_summary and solution_summary.lower() not in ("not ready yet", "no summary", ""):
                    print(f"   Attempting to conclude with available summary...")
                    return self._create_conclude_step(session, query, step, solution_summary)
            return None

    def print_plan_version(self, session: AgentSession, version_num: int) -> None:
        """Print the current plan version."""
        if not session.plan_versions:
            return
        current_plan = session.plan_versions[-1]
        print(f"\n[Decision Plan Text: V{version_num}]:")
        if current_plan.reason:
            print(f"  Reason: {current_plan.reason}")
        for line in current_plan.plan_text:
            print(f"  {line}")
