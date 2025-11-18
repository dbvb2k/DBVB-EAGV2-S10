import os
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError, APIError
import re
from mcp_servers.multiMCP import MultiMCP
from config.settings import get_config


load_dotenv()


class Decision:
    """
    Decision module for generating execution plans based on perception and query.
    
    Handles LLM communication with retry logic, error recovery, and response validation.
    """
    
    def __init__(
        self, 
        decision_prompt_path: str, 
        multi_mcp: MultiMCP, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
        request_timeout: Optional[float] = None
    ):
        """
        Initialize Decision module.
        
        Args:
            decision_prompt_path: Path to decision prompt template
            multi_mcp: MultiMCP instance for tool descriptions
            api_key: Optional API key (uses env var if not provided)
            model: Optional model name (uses config if not provided)
            max_retries: Optional max retries (uses config if not provided)
            request_timeout: Optional timeout (uses config if not provided)
        """
        load_dotenv()
        
        # Load configuration
        config = get_config().get_llm_config("decision")
        
        self.decision_prompt_path = decision_prompt_path
        self.multi_mcp = multi_mcp
        self.model = model or config.model
        self.max_retries = max_retries or config.max_retries
        self.request_timeout = request_timeout or config.request_timeout
        self.initial_retry_delay = config.initial_retry_delay
        self.max_retry_delay = config.max_retry_delay

        self.api_key = api_key or config.api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or configuration.")
        self.client = genai.Client(api_key=self.api_key)

    def run(self, decision_input: dict) -> dict:
        """
        Generate decision/plan based on input.
        
        Args:
            decision_input: Input dictionary with query, perception, etc.
            
        Returns:
            Decision output dictionary with plan and step information
        """
        prompt_template = Path(self.decision_prompt_path).read_text(encoding="utf-8")
        function_list_text = self.multi_mcp.tool_description_wrapper()
        tool_descriptions = "\n".join(f"- `{desc.strip()}`" for desc in function_list_text)
        tool_descriptions = "\n\n### The ONLY Available Tools\n\n---\n\n" + tool_descriptions
        full_prompt = f"{prompt_template.strip()}\n{tool_descriptions}\n\n```json\n{json.dumps(decision_input, indent=2)}\n```"

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm_with_timeout(full_prompt)
                raw_text = self._extract_response_text(response)
                return self._parse_response(raw_text, decision_input)
                
            except (ServerError, APIError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    print(f"⚠️ Decision LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"🚫 Decision LLM failed after {self.max_retries} attempts: {e}")
                    return self._create_error_response(
                        "Decision model unavailable after retries.",
                        f"Server error: {str(e)}"
                    )
                    
            except TimeoutError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    print(f"⚠️ Decision LLM timeout (attempt {attempt + 1}/{self.max_retries})")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"🚫 Decision LLM timed out after {self.max_retries} attempts")
                    return self._create_error_response(
                        "Decision model request timed out.",
                        f"Timeout after {self.request_timeout}s"
                    )
                    
            except Exception as e:
                last_exception = e
                print(f"❌ Unexpected error in Decision module: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    time.sleep(delay)
                else:
                    return self._create_error_response(
                        "Unexpected error in decision generation.",
                        f"{type(e).__name__}: {str(e)}"
                    )

        # Fallback if all retries exhausted
        return self._create_error_response(
            "Decision generation failed after all retries.",
            str(last_exception) if last_exception else "Unknown error"
        )

    def _call_llm_with_timeout(self, prompt: str):
        """Call LLM with timeout protection."""
        # Note: genai.Client doesn't have built-in timeout, so we use asyncio
        # For synchronous calls, we'll rely on the library's timeout if available
        # Otherwise, we catch timeout exceptions from the underlying HTTP client
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

    def _parse_response(self, raw_text: str, decision_input: dict) -> dict:
        """
        Parse LLM response with multiple fallback strategies.
        
        Args:
            raw_text: Raw text from LLM
            decision_input: Original input for context
            
        Returns:
            Parsed decision output dictionary
        """
        # Strategy 1: Try to find JSON block with regex
        json_block = self._extract_json_block(raw_text)
        
        if json_block:
            try:
                output = json.loads(json_block)
                return self._validate_and_normalize_output(output)
            except json.JSONDecodeError:
                # Strategy 2: Try to salvage partial JSON
                return self._salvage_partial_json(json_block, raw_text)
        
        # Strategy 3: Try to find JSON anywhere in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                output = json.loads(json_match.group(0))
                return self._validate_and_normalize_output(output)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Return error response
        print("⚠️ Could not parse JSON from LLM response")
        return self._create_error_response(
            "Could not parse decision response from LLM.",
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

    def _salvage_partial_json(self, json_block: str, raw_text: str) -> dict:
        """Attempt to salvage partial JSON by extracting key fields."""
        print("⚠️ JSON decode failed, attempting to salvage partial data...")
        
        salvaged = {
            "step_index": 0,
            "description": "Recovered partial JSON from LLM.",
            "type": "NOP",
            "code": "",
            "conclusion": "",
            "plan_text": ["Step 0: Partial plan recovered due to JSON decode error."],
            "raw_text": raw_text[:1000]
        }
        
        # Try to extract code field
        code_match = re.search(r'code\s*[:=]\s*"(.*?)"', json_block, re.DOTALL)
        if code_match:
            try:
                code_value = bytes(code_match.group(1), "utf-8").decode("unicode_escape")
                salvaged["code"] = code_value
                salvaged["type"] = "CODE"
            except Exception:
                pass
        
        # Try to extract description
        desc_match = re.search(r'description\s*[:=]\s*"(.*?)"', json_block, re.DOTALL)
        if desc_match:
            salvaged["description"] = desc_match.group(1)
        
        # Try to extract type
        type_match = re.search(r'type\s*[:=]\s*"(.*?)"', json_block, re.DOTALL)
        if type_match:
            salvaged["type"] = type_match.group(1).upper()
        
        return salvaged

    def _validate_and_normalize_output(self, output: dict) -> dict:
        """Validate and normalize decision output."""
        # Handle flattened or nested format
        if "next_step" in output:
            output.update(output.pop("next_step"))
        
        # Ensure required fields with defaults
        defaults = {
            "step_index": 0,
            "description": "Missing from LLM response",
            "type": "NOP",
            "code": "",
            "conclusion": "",
            "plan_text": ["Step 0: No valid plan returned by LLM."]
        }
        
        for key, default in defaults.items():
            output.setdefault(key, default)
        
        # Validate step type
        valid_types = {"CODE", "CONCLUDE", "NOP", "NOOP"}
        if output["type"] not in valid_types:
            print(f"⚠️ Invalid step type '{output['type']}', defaulting to NOP")
            output["type"] = "NOP"
        
        # Ensure plan_text is a list
        if isinstance(output.get("plan_text"), str):
            output["plan_text"] = [output["plan_text"]]
        elif not isinstance(output.get("plan_text"), list):
            output["plan_text"] = defaults["plan_text"]
        
        return output

    def _create_error_response(self, description: str, error_detail: str = "") -> dict:
        """Create a standardized error response."""
        return {
            "step_index": 0,
            "description": description,
            "type": "NOP",
            "code": "",
            "conclusion": "",
            "plan_text": [f"Step 0: {description}"],
            "error": error_detail,
            "raw_text": error_detail[:1000] if error_detail else ""
        }
