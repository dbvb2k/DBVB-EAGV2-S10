import asyncio
import yaml
from mcp_servers.multiMCP import MultiMCP

from dotenv import load_dotenv
from agent.agent_loop import AgentLoop
from pprint import pprint
BANNER = """
──────────────────────────────────────────────────────
🔸  Agentic Query Assistant  🔸
Type your question and press Enter.
Type 'exit' or 'quit' to leave.
──────────────────────────────────────────────────────
"""


async def interactive() -> None:
    print(BANNER)
    print("Loading MCP Servers...")
    with open("config/mcp_server_config.yaml", "r") as f:
        profile = yaml.safe_load(f)
        mcp_servers_list = profile.get("mcp_servers", [])
        configs = list(mcp_servers_list)

    # Initialize MCP + Dispatcher
    multi_mcp = MultiMCP(server_configs=configs)
    await multi_mcp.initialize()
    loop = AgentLoop(
        perception_prompt_path="prompts/perception_prompt.txt",
        decision_prompt_path="prompts/decision_prompt.txt",
        multi_mcp=multi_mcp,
        strategy="exploratory"
    )
    while True:

        query = input("🟢  You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break


        response = await loop.run(query)
        # response = await loop.run("What is 4 + 4?")
        # pprint(f"🔵  Agent: {response.state.final_answer}\n {response.state.reasoning_note}\n")
        # Use attribute access since state is now a SessionState Pydantic model
        solution = response.state.solution_summary or response.state.final_answer or "No solution provided"
        print(f"🔵 Agent: {solution}\n")

        follow = input("\n\nContinue? (press Enter) or type 'exit': ").strip()
        if follow.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break

if __name__ == "__main__":
    asyncio.run(interactive())
