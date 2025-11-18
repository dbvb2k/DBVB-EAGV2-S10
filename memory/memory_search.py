import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from rapidfuzz import fuzz
import aiofiles
import aiofiles.os


class MemorySearch:
    """
    Async memory search for retrieving relevant past sessions.
    
    Uses async file I/O to prevent blocking the event loop.
    """
    
    def __init__(self, logs_path: str = "memory/session_logs"):
        self.logs_path = Path(logs_path)

    async def search_memory(self, user_query: str, top_k: int = 3) -> List[Dict]:
        """
        Search memory for relevant past sessions.
        
        Args:
            user_query: Query string to search for
            top_k: Number of top results to return
            
        Returns:
            List of matching memory entries
        """
        memory_entries = await self._load_queries()
        scored_results = []

        for entry in memory_entries:
            query_score = fuzz.partial_ratio(user_query.lower(), entry["query"].lower())
            summary_score = fuzz.partial_ratio(user_query.lower(), entry["solution_summary"].lower())
            length_penalty = len(entry["solution_summary"]) / 100
            score = 0.5 * query_score + 0.4 * summary_score - 0.05 * length_penalty
            scored_results.append((score, entry))

        top_matches = sorted(scored_results, key=lambda x: x[0], reverse=True)[:top_k]
        return [match[1] for match in top_matches]

    async def _load_queries(self) -> List[Dict]:
        """Load and parse all JSON files from the logs directory."""
        memory_entries = []
        
        # Get all JSON files (async path operations)
        try:
            all_json_files = [
                f for f in self.logs_path.rglob("*.json")
                if await aiofiles.os.path.isfile(f)
            ]
        except Exception as e:
            print(f"⚠️ Error scanning directory '{self.logs_path}': {e}")
            return memory_entries

        print(f"🔍 Found {len(all_json_files)} JSON file(s) in '{self.logs_path}'")

        # Process files concurrently (with limit to avoid overwhelming system)
        semaphore = asyncio.Semaphore(10)  # Process up to 10 files concurrently
        
        async def process_file(file: Path):
            async with semaphore:
                count_before = len(memory_entries)
                try:
                    async with aiofiles.open(file, 'r', encoding='utf-8') as f:
                        content_str = await f.read()
                        content = json.loads(content_str)

                    if isinstance(content, list):  # FORMAT 1
                        for session in content:
                            await self._extract_entry(session, file.name, memory_entries)
                    elif isinstance(content, dict) and "session_id" in content:  # FORMAT 2
                        await self._extract_entry(content, file.name, memory_entries)
                    elif isinstance(content, dict) and "turns" in content:  # FORMAT 3
                        for turn in content["turns"]:
                            await self._extract_entry(turn, file.name, memory_entries)

                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping '{file}' (invalid JSON): {e}")
                except Exception as e:
                    print(f"⚠️ Skipping '{file}': {e}")

                count_after = len(memory_entries)
                if count_after > count_before:
                    print(f"✅ {file.name}: {count_after - count_before} matching entries")

        # Process all files concurrently
        await asyncio.gather(*[process_file(file) for file in all_json_files])

        print(f"📦 Total usable memory entries collected: {len(memory_entries)}\n")
        return memory_entries

    async def _extract_entry(self, obj: dict, file_name: str, memory_entries: List[Dict]):
        """
        Extract memory entry from session object.
        
        This is async to maintain consistency, though the actual work is synchronous.
        """
        original_obj = obj  # keep top-level reference

        def recursive_find(obj: dict) -> Optional[dict]:
            """Recursively find entries with original_goal_achieved=True."""
            if isinstance(obj, dict):
                if obj.get("original_goal_achieved") is True:
                    query = extract_query(original_obj)  # pull from full session object
                    return {
                        "query": query,
                        "summary": obj.get("solution_summary", ""),
                        "requirement": obj.get("result_requirement", "")
                    }
                for v in obj.values():
                    result = recursive_find(v)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = recursive_find(item)
                    if result:
                        return result
            return None

        def extract_query(obj: dict) -> str:
            """Recursively extract query string from object."""
            if isinstance(obj, dict):
                if "query" in obj and isinstance(obj["query"], str):
                    return obj["query"]
                for v in obj.values():
                    q = extract_query(v)
                    if q:
                        return q
            elif isinstance(obj, list):
                for item in obj:
                    q = extract_query(item)
                    if q:
                        return q
            return ""

        try:
            match = recursive_find(obj)
            if match and match["query"]:
                print(f"✅ Extracted: {match['query'][:40]} → {match['summary'][:40]}")
                memory_entries.append({
                    "file": file_name,
                    "query": match["query"],
                    "result_requirement": match["requirement"],
                    "solution_summary": match["summary"]
                })
        except Exception as e:
            print(f"❌ Error parsing {file_name}: {e}")


# Synchronous wrapper for backward compatibility
def search_memory_sync(user_query: str, top_k: int = 3) -> List[Dict]:
    """
    Synchronous wrapper for search_memory.
    
    For use in non-async contexts. Creates a new event loop if needed.
    """
    searcher = MemorySearch()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(searcher.search_memory(user_query, top_k))
        else:
            return loop.run_until_complete(searcher.search_memory(user_query, top_k))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(searcher.search_memory(user_query, top_k))


if __name__ == "__main__":
    async def main():
        searcher = MemorySearch()
        query = input("Enter your query: ").strip()
        results = await searcher.search_memory(query)

        if not results:
            print("❌ No matching memory entries found.")
        else:
            print("\n🎯 Top Matches:\n")
            for i, res in enumerate(results, 1):
                print(f"[{i}] File: {res['file']}\nQuery: {res['query']}\nResult Requirement: {res['result_requirement']}\nSummary: {res['solution_summary']}\n")
    
    asyncio.run(main())
