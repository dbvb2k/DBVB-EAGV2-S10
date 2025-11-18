import os
import json
import re
import time
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, APIError
from config.settings import get_config


load_dotenv()


class Perception:
    """
    Perception module for analyzing inputs and determining goal achievement status.
    
    Handles LLM communication with retry logic, error recovery, and response validation.
    """
    
    def __init__(
        self, 
        perception_prompt_path: str, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None
    ):
        """
        Initialize Perception module.
        
        Args:
            perception_prompt_path: Path to perception prompt template
            api_key: Optional API key (uses env var if not provided)
            model: Optional model name (uses config if not provided)
            max_retries: Optional max retries (uses config if not provided)
            request_timeout: Optional timeout (uses config if not provided)
        """
        load_dotenv()
        
        # Load configuration
        config = get_config().get_llm_config("perception")
        
        self.api_key = api_key or config.api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or configuration.")
        self.client = genai.Client(api_key=self.api_key)
        self.perception_prompt_path = perception_prompt_path
        self.model = model or config.model
        self.max_retries = max_retries or config.max_retries
        self.request_timeout = request_timeout or config.request_timeout
        self.initial_retry_delay = config.initial_retry_delay
        self.max_retry_delay = config.max_retry_delay

    def build_perception_input(
        self, 
        raw_input: str, 
        memory: List[Dict[str, Any]], 
        current_plan: str = "", 
        snapshot_type: str = "user_query"
    ) -> dict:
        """
        Build perception input dictionary.
        
        Args:
            raw_input: Raw input text to analyze
            memory: List of memory entries
            current_plan: Current execution plan
            snapshot_type: Type of perception snapshot
            
        Returns:
            Perception input dictionary
        """
        if memory:
            memory_excerpt = {
                f"memory_{i+1}": {
                    "query": res.get("query", ""),
                    "result_requirement": res.get("result_requirement", ""),
                    "solution_summary": res.get("solution_summary", "")
                }
                for i, res in enumerate(memory)
            }
        else:
            memory_excerpt = {}

        return {
            "run_id": str(uuid.uuid4()),
            "snapshot_type": snapshot_type,
            "raw_input": raw_input,
            "memory_excerpt": memory_excerpt,
            "prev_objective": "",
            "prev_confidence": None,
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "schema_version": 1,
            "current_plan": current_plan or "Initial Query Mode, plan not created"
        }
    
    def run(self, perception_input: dict) -> dict:
        """
        Run perception on given input using the specified prompt file.
        
        Args:
            perception_input: Input dictionary
            
        Returns:
            Perception result dictionary
        """
        prompt_template = Path(self.perception_prompt_path).read_text(encoding="utf-8")
        full_prompt = f"{prompt_template.strip()}\n\n```json\n{json.dumps(perception_input, indent=2)}\n```"

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm_with_timeout(full_prompt)
                raw_text = self._extract_response_text(response)
                return self._parse_response(raw_text)
                
            except (ServerError, APIError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    print(f"⚠️ Perception LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"🚫 Perception LLM failed after {self.max_retries} attempts: {e}")
                    return self._create_error_response(
                        "Perception model unavailable after retries.",
                        f"Server error: {str(e)}"
                    )
                    
            except TimeoutError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    print(f"⚠️ Perception LLM timeout (attempt {attempt + 1}/{self.max_retries})")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"🚫 Perception LLM timed out after {self.max_retries} attempts")
                    return self._create_error_response(
                        "Perception model request timed out.",
                        f"Timeout after {self.request_timeout}s"
                    )
                    
            except Exception as e:
                last_exception = e
                print(f"❌ Unexpected error in Perception module: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    time.sleep(delay)
                else:
                    return self._create_error_response(
                        "Unexpected error in perception processing.",
                        f"{type(e).__name__}: {str(e)}"
                    )

        # Fallback if all retries exhausted
        return self._create_error_response(
            "Perception processing failed after all retries.",
            str(last_exception) if last_exception else "Unknown error"
        )

    def _call_llm_with_timeout(self, prompt: str):
        """Call LLM with timeout protection."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response
        except Exception as e:
            # Check if it's a timeout-related error
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                raise TimeoutError(f"Request timed out: {e}")
            raise

    def _extract_response_text(self, response) -> str:
        """Extract text from LLM response, handling different response formats."""
        try:
            # Try primary method
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # Try candidates method
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, 'text'):
                            return part.text.strip()
            
            # Fallback to string representation
            return str(response).strip()
            
        except (AttributeError, IndexError, KeyError) as e:
            raise ValueError(f"Unable to extract text from response: {e}")

    def _parse_response(self, raw_text: str) -> dict:
        """
        Parse LLM response with multiple fallback strategies.
        
        Args:
            raw_text: Raw text from LLM
            
        Returns:
            Parsed perception result dictionary
        """
        # Strategy 1: Try to find JSON block
        json_block = self._extract_json_block(raw_text)
        
        if json_block:
            try:
                output = json.loads(json_block)
                return self._validate_and_normalize_output(output)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode failed: {e}")
                # Try to salvage partial JSON
                return self._salvage_partial_json(json_block)
        
        # Strategy 2: Try to find JSON anywhere in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                output = json.loads(json_match.group(0))
                return self._validate_and_normalize_output(output)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Return error response
        print("⚠️ Could not parse JSON from Perception LLM response")
        return self._create_error_response(
            "Could not parse perception response from LLM.",
            f"Response preview: {raw_text[:200]}..."
        )

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extract JSON block from text using multiple patterns."""
        patterns = [
            r"```json\s*(\{.*?\})\s*```",  # Standard markdown JSON block
            r"```\s*(\{.*?\})\s*```",       # Markdown code block without language
            r"(\{[\s\S]*\})",                # Any JSON object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None

    def _salvage_partial_json(self, json_block: str) -> dict:
        """Attempt to salvage partial JSON by extracting key fields."""
        salvaged = {
            "entities": [],
            "result_requirement": "N/A",
            "original_goal_achieved": False,
            "reasoning": "Partial JSON recovered from LLM response.",
            "local_goal_achieved": False,
            "local_reasoning": "Could not fully parse response.",
            "last_tooluse_summary": "None",
            "solution_summary": "Not ready yet",
            "confidence": "0.0"
        }
        
        # Try to extract key fields using regex
        field_patterns = {
            "original_goal_achieved": r'original_goal_achieved\s*[:=]\s*(true|false)',
            "local_goal_achieved": r'local_goal_achieved\s*[:=]\s*(true|false)',
            "confidence": r'confidence\s*[:=]\s*"?([0-9.]+)"?',
            "solution_summary": r'solution_summary\s*[:=]\s*"(.*?)"',
            "reasoning": r'reasoning\s*[:=]\s*"(.*?)"',
        }
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, json_block, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1)
                if field in ["original_goal_achieved", "local_goal_achieved"]:
                    salvaged[field] = value.lower() == "true"
                elif field == "confidence":
                    try:
                        salvaged[field] = str(float(value))
                    except ValueError:
                        pass
                else:
                    salvaged[field] = value[:500]  # Limit length
        
        return salvaged

    def _validate_and_normalize_output(self, output: dict) -> dict:
        """Validate and normalize perception output."""
        # Ensure required fields with defaults
        required_fields = {
            "entities": [],
            "result_requirement": "No requirement specified.",
            "original_goal_achieved": False,
            "reasoning": "No reasoning given.",
            "local_goal_achieved": False,
            "local_reasoning": "No local reasoning given.",
            "last_tooluse_summary": "None",
            "solution_summary": "No summary.",
            "confidence": "0.0"
        }
        
        for key, default in required_fields.items():
            output.setdefault(key, default)
        
        # Ensure entities is a list
        if not isinstance(output["entities"], list):
            output["entities"] = []
        
        # Ensure boolean fields are actually booleans
        for bool_field in ["original_goal_achieved", "local_goal_achieved"]:
            if not isinstance(output[bool_field], bool):
                output[bool_field] = str(output[bool_field]).lower() in ("true", "1", "yes")
        
        # Ensure confidence is a string representation of float
        try:
            conf_value = float(output["confidence"])
            output["confidence"] = str(max(0.0, min(1.0, conf_value)))
        except (ValueError, TypeError):
            output["confidence"] = "0.0"
        
        return output

    def _create_error_response(self, reasoning: str, error_detail: str = "") -> dict:
        """Create a standardized error response."""
        return {
            "entities": [],
            "result_requirement": "N/A",
            "original_goal_achieved": False,
            "reasoning": reasoning,
            "local_goal_achieved": False,
            "local_reasoning": f"Error: {error_detail}" if error_detail else "Could not extract structured information.",
            "last_tooluse_summary": "None",
            "solution_summary": "Not ready yet",
            "confidence": "0.0",
            "error": error_detail
        }
