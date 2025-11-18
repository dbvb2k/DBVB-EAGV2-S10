import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
import aiofiles
import aiofiles.os


async def get_store_path(session_id: str, base_dir: str = "memory/session_logs") -> Path:
    """
    Construct the full path to the session file based on current date and session ID.
    Format: memory/session_logs/YYYY/MM/DD/<session_id>.json
    
    Args:
        session_id: Session identifier
        base_dir: Base directory for session logs
        
    Returns:
        Path to session file
    """
    now = datetime.now()
    day_dir = Path(base_dir) / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
    
    # Create directory asynchronously
    try:
        await aiofiles.os.makedirs(day_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not create directory {day_dir}: {e}")
        # Fallback to synchronous mkdir
        day_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{session_id}.json"
    return day_dir / filename


def simplify_session_id(session_id: str) -> str:
    """
    Return the simplified (short) version of the session ID for display/logging.
    
    Args:
        session_id: Full session ID
        
    Returns:
        Shortened session ID
    """
    return session_id.split("-")[0]


async def append_session_to_store(session_obj, base_dir: str = "memory/session_logs") -> None:
    """
    Save the session object as a standalone file asynchronously.
    
    If a file already exists and is corrupt, it will be overwritten with fresh data.
    
    Args:
        session_obj: Session object with to_json() method
        base_dir: Base directory for session logs
    """
    session_data = session_obj.to_json()
    session_data["_session_id_short"] = simplify_session_id(session_data["session_id"])

    store_path = await get_store_path(session_data["session_id"], base_dir)

    # Check if file exists and validate JSON
    if await aiofiles.os.path.exists(store_path):
        try:
            async with aiofiles.open(store_path, "r", encoding="utf-8") as f:
                existing = await f.read()
                if existing.strip():
                    json.loads(existing)  # verify valid JSON
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Corrupt JSON detected in {store_path}. Overwriting.")
        except Exception as e:
            print(f"⚠️ Warning: Error reading existing file {store_path}: {e}")

    # Write session data
    try:
        async with aiofiles.open(store_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(session_data, indent=2))
        print(f"✅ Session stored: {store_path}")
    except Exception as e:
        print(f"❌ Failed to write session to {store_path}: {e}")
        raise


async def live_update_session(session_obj, base_dir: str = "memory/session_logs") -> None:
    """
    Update (or overwrite) the session file with latest data asynchronously.
    
    In per-file format, this is identical to append.
    
    Args:
        session_obj: Session object with to_json() method
        base_dir: Base directory for session logs
    """
    try:
        await append_session_to_store(session_obj, base_dir)
        print("📝 Session live-updated.")
    except Exception as e:
        print(f"❌ Failed to update session: {e}")


# Synchronous wrappers for backward compatibility
def append_session_to_store_sync(session_obj, base_dir: str = "memory/session_logs") -> None:
    """Synchronous wrapper for append_session_to_store."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(append_session_to_store(session_obj, base_dir))
        else:
            loop.run_until_complete(append_session_to_store(session_obj, base_dir))
    except RuntimeError:
        asyncio.run(append_session_to_store(session_obj, base_dir))


def live_update_session_sync(session_obj, base_dir: str = "memory/session_logs") -> None:
    """Synchronous wrapper for live_update_session."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(live_update_session(session_obj, base_dir))
        else:
            loop.run_until_complete(live_update_session(session_obj, base_dir))
    except RuntimeError:
        asyncio.run(live_update_session(session_obj, base_dir))
