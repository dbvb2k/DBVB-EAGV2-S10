import uuid
import json
from typing import Optional
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
            session.state.update({
                "original_goal_achieved": False,
                "final_answer": "Execution stopped due to maximum iteration limit.",
                "confidence": 0.0,
                "reasoning_note": "Agent loop exceeded maximum iterations.",
                "solution_summary": "Unable to complete task within iteration limits."
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

    def run_perception(
        self, 
        query: str, 
        memory_results: list, 
        session_memory: Optional[list] = None,
        snapshot_type: str = "user_query",
        current_plan: Optional[list] = None
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
            snapshot_type=snapshot_type
        )
        perception_result = self.perception.run(perception_input)
        
        print(f"\n[Perception Result ({snapshot_type})]:")
        print(json.dumps(perception_result, indent=2, ensure_ascii=False))
        
        return perception_result

    async def handle_perception_completion(self, session: AgentSession, perception_result: dict) -> None:
        """Handle early completion when perception is confident."""
        print("\n✅ Perception has already fully answered the query.")
        session.state.update({
            "original_goal_achieved": True,
            "final_answer": perception_result.get("solution_summary", "Answer ready."),
            "confidence": perception_result.get("confidence", 0.95),
            "reasoning_note": perception_result.get("reasoning", "Fully handled by initial perception."),
            "solution_summary": perception_result.get("solution_summary", "Answer ready.")
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
        """Create a Step object from decision output."""
        return Step(
            index=decision_output["step_index"],
            description=decision_output["description"],
            type=decision_output["type"],  # Will be normalized by Pydantic validator
            code=ToolCode(
                tool_name="raw_code_block",
                tool_arguments={"code": decision_output["code"]}
            ) if decision_output.get("type", "").upper() in ("CODE", StepType.CODE.value) else None,
            conclusion=decision_output.get("conclusion"),
        )

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

        if step.type == StepType.CODE:
            return await self._execute_code_step(step, session, session_memory)
        elif step.type == StepType.CONCLUDE:
            return await self._execute_conclude_step(step, session, session_memory)
        elif step.type in (StepType.NOP, StepType.NOOP):
            return await self._execute_nop_step(step, session)
        else:
            print(f"⚠️ Unknown step type: {step.type}")
            return step.mark_failed(f"Unknown step type: {step.type}")

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
            perception_result = self.run_perception(
                query=executor_response.get('result', 'Tool Failed'),
                memory_results=session_memory,
                current_plan=current_plan_text,
                snapshot_type="step_result"
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

        perception_result = self.run_perception(
            query=step.conclusion,
            memory_results=session_memory,
            current_plan=session.plan_versions[-1].plan_text if session.plan_versions else [],
            snapshot_type="step_result"
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
            # Proceed to next step
            return self.get_next_step(session, query, step)
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
                "current_step": step.to_dict()
            })
            
            next_step = session.add_plan_version(
                decision_output["plan_text"],
                [self.create_step(decision_output)]
            )
            self.print_plan_version(session, len(session.plan_versions))
            return next_step
        else:
            print("\n✅ No more steps in current plan.")
            return None

    def _replan(self, session: AgentSession, query: str, step: Step) -> Optional[Step]:
        """Generate a new plan after step failure."""
        if not session.plan_versions:
            return None
        current_plan = session.plan_versions[-1]
        
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
            "current_step": step.to_dict()
        })
        
        new_step = session.add_plan_version(
            decision_output["plan_text"],
            [self.create_step(decision_output)],
            reason=f"Replanned after step {step.index} failure"
        )
        self.print_plan_version(session, len(session.plan_versions))
        return new_step

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
