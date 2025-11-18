import ast
import asyncio
import time
import builtins
import textwrap
import traceback
from datetime import datetime
from typing import Dict, Any, Optional


# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
from config.settings import get_config

# Get configuration (will be loaded on first import)
_config_manager = get_config()
_executor_config = _config_manager.get_executor_config()
ALLOWED_MODULES = set(_executor_config.allowed_modules)
MAX_FUNCTIONS = _executor_config.max_functions
TIMEOUT_PER_FUNCTION = _executor_config.timeout_per_function
MIN_TIMEOUT = _executor_config.min_timeout
MAX_AST_DEPTH = _executor_config.max_ast_depth


class ExecutorError(Exception):
    """Base exception for executor errors."""
    pass


class CodeValidationError(ExecutorError):
    """Raised when code validation fails."""
    pass


class ExecutionTimeoutError(ExecutorError):
    """Raised when code execution times out."""
    pass


class KeywordStripper(ast.NodeTransformer):
    """Rewrite all function calls to remove keyword args and keep only values as positional."""
    
    def visit_Call(self, node):
        self.generic_visit(node)
        if node.keywords:
            # Convert all keyword arguments into positional args (discard names)
            for kw in node.keywords:
                node.args.append(kw.value)
            node.keywords = []
        return node


class AwaitTransformer(ast.NodeTransformer):
    """AST transformer: auto-await known async MCP tools."""
    
    def __init__(self, async_funcs):
        self.async_funcs = async_funcs

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.async_funcs:
            return ast.Await(value=node)
        return node


class ASTDepthValidator(ast.NodeVisitor):
    """Validate AST depth to prevent stack overflow attacks."""
    
    def __init__(self, max_depth: int = MAX_AST_DEPTH):
        self.max_depth = max_depth
        self.current_depth = 0
        self.max_seen_depth = 0
    
    def visit(self, node):
        self.current_depth += 1
        self.max_seen_depth = max(self.max_seen_depth, self.current_depth)
        
        if self.current_depth > self.max_depth:
            raise CodeValidationError(
                f"AST depth ({self.current_depth}) exceeds maximum ({self.max_depth}). "
                "This may indicate a malicious or overly complex code structure."
            )
        
        try:
            self.generic_visit(node)
        finally:
            self.current_depth -= 1


# ───────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ───────────────────────────────────────────────────────────────
def count_function_calls(code: str) -> int:
    """Count function calls in code."""
    try:
        tree = ast.parse(code)
        return sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    except SyntaxError as e:
        raise CodeValidationError(f"Syntax error in code: {e}")


def validate_ast(code: str) -> None:
    """Validate AST structure and depth."""
    try:
        tree = ast.parse(code)
        validator = ASTDepthValidator()
        validator.visit(tree)
    except SyntaxError as e:
        raise CodeValidationError(f"Syntax error: {e}")
    except RecursionError:
        raise CodeValidationError("Code structure too deep (possible recursion attack)")


def build_safe_globals(mcp_funcs: dict, multi_mcp=None) -> dict:
    """Build safe globals dictionary for code execution."""
    safe_globals = {
        "__builtins__": {
            k: getattr(builtins, k)
            for k in ("range", "len", "int", "float", "str", "list", "dict", "print", "sum", "__import__")
        },
        **mcp_funcs,
    }

    # Import allowed modules
    for module in ALLOWED_MODULES:
        try:
            safe_globals[module] = __import__(module)
        except ImportError:
            # Skip modules that aren't available
            pass

    # Store LLM-style result
    safe_globals["final_answer"] = lambda x: safe_globals.setdefault("result_holder", x)

    # Optional: add parallel execution
    if multi_mcp:
        async def parallel(*tool_calls):
            """Execute multiple tool calls in parallel."""
            coros = [
                multi_mcp.function_wrapper(tool_name, *args)
                for tool_name, *args in tool_calls
            ]
            return await asyncio.gather(*coros)

        safe_globals["parallel"] = parallel

    return safe_globals


# ───────────────────────────────────────────────────────────────
# MAIN EXECUTOR
# ───────────────────────────────────────────────────────────────
async def run_user_code(code: str, multi_mcp) -> dict:
    """
    Execute user-provided code in a sandboxed environment.
    
    Args:
        code: Python code string to execute
        multi_mcp: MultiMCP instance for tool access
        
    Returns:
        Dictionary with status, result/error, and timing information
    """
    start_time = time.perf_counter()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    func_count = 0
    
    try:
        # Step 1: Validate code structure
        try:
            func_count = count_function_calls(code)
            validate_ast(code)
        except CodeValidationError as e:
            return _create_error_response(
                start_timestamp,
                start_time,
                f"Code validation failed: {str(e)}",
                error_type="validation_error"
            )
        
        # Step 2: Check function count limit
        if func_count > MAX_FUNCTIONS:
            return _create_error_response(
                start_timestamp,
                start_time,
                f"Too many function calls ({func_count} > {MAX_FUNCTIONS})",
                error_type="limit_exceeded"
            )

        # Step 3: Build execution environment
        try:
            tool_funcs = {
                tool.name: make_tool_proxy(tool.name, multi_mcp)
                for tool in multi_mcp.get_all_tools()
            }
            sandbox = build_safe_globals(tool_funcs, multi_mcp)
            local_vars = {}
        except Exception as e:
            return _create_error_response(
                start_timestamp,
                start_time,
                f"Failed to build execution environment: {str(e)}",
                error_type="setup_error"
            )

        # Step 4: Parse and transform code
        try:
            cleaned_code = textwrap.dedent(code.strip())
            tree = ast.parse(cleaned_code)
            
            # Check for return statement or result assignment
            has_return = any(isinstance(node, ast.Return) for node in tree.body)
            has_result = any(
                isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "result" for t in node.targets
                )
                for node in tree.body
            )
            
            # Add return statement if result is assigned but not returned
            if not has_return and has_result:
                tree.body.append(ast.Return(value=ast.Name(id="result", ctx=ast.Load())))

            # Apply AST transformations
            tree = KeywordStripper().visit(tree)
            tree = AwaitTransformer(set(tool_funcs)).visit(tree)
            ast.fix_missing_locations(tree)

            # Wrap in async function
            func_def = ast.AsyncFunctionDef(
                name="__main",
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=tree.body,
                decorator_list=[]
            )
            wrapper = ast.Module(body=[func_def], type_ignores=[])
            ast.fix_missing_locations(wrapper)

            compiled = compile(wrapper, filename="<user_code>", mode="exec")
        except SyntaxError as e:
            return _create_error_response(
                start_timestamp,
                start_time,
                f"Syntax error in code: {str(e)}",
                error_type="syntax_error"
            )
        except Exception as e:
            return _create_error_response(
                start_timestamp,
                start_time,
                f"Code transformation failed: {type(e).__name__}: {str(e)}",
                error_type="transformation_error"
            )

        # Step 5: Execute code
        try:
            exec(compiled, sandbox, local_vars)
            
            # Calculate timeout
            timeout = max(MIN_TIMEOUT, func_count * TIMEOUT_PER_FUNCTION)
            
            # Execute with timeout
            try:
                returned = await asyncio.wait_for(
                    local_vars["__main"](), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise ExecutionTimeoutError(
                    f"Execution timed out after {timeout} seconds "
                    f"(calculated from {func_count} function calls)"
                )

            # Extract result
            result_value = returned if returned is not None else sandbox.get("result_holder", None)

            # Handle tool errors
            if result_value is not None and hasattr(result_value, "isError"):
                if getattr(result_value, "isError", False):
                    error_msg = _extract_error_message(result_value)
                    return _create_error_response(
                        start_timestamp,
                        start_time,
                        f"Tool execution error: {error_msg}",
                        error_type="tool_error"
                    )

            # Success response
            return {
                "status": "success",
                "result": str(result_value) if result_value is not None else "None",
                "execution_time": start_timestamp,
                "total_time": str(round(time.perf_counter() - start_time, 3)),
                "function_calls": func_count
            }

        except ExecutionTimeoutError as e:
            return _create_error_response(
                start_timestamp,
                start_time,
                str(e),
                error_type="timeout_error"
            )
        except Exception as e:
            # Capture full traceback for debugging
            error_trace = traceback.format_exc()
            return _create_error_response(
                start_timestamp,
                start_time,
                f"{type(e).__name__}: {str(e)}",
                error_type="execution_error",
                traceback=error_trace
            )

    except Exception as e:
        # Catch-all for unexpected errors
        error_trace = traceback.format_exc()
        return _create_error_response(
            start_timestamp,
            start_time,
            f"Unexpected error: {type(e).__name__}: {str(e)}",
            error_type="unexpected_error",
            traceback=error_trace
        )


def _create_error_response(
    start_timestamp: str,
    start_time: float,
    error_message: str,
    error_type: str = "error",
    traceback: Optional[str] = None
) -> dict:
    """Create standardized error response."""
    response = {
        "status": "error",
        "error": error_message,
        "error_type": error_type,
        "execution_time": start_timestamp,
        "total_time": str(round(time.perf_counter() - start_time, 3))
    }
    
    # Include traceback in debug mode (could be controlled by config)
    if traceback:
        response["traceback"] = traceback
    
    return response


def _extract_error_message(result_value: Any) -> str:
    """Extract error message from tool result."""
    try:
        if hasattr(result_value, "content") and result_value.content:
            if isinstance(result_value.content, list) and len(result_value.content) > 0:
                content_item = result_value.content[0]
                if hasattr(content_item, "text"):
                    return content_item.text.strip()
        return str(result_value)
    except Exception:
        return "Unknown tool error"


# ───────────────────────────────────────────────────────────────
# TOOL WRAPPER
# ───────────────────────────────────────────────────────────────
def make_tool_proxy(tool_name: str, mcp):
    """Create a proxy function for an MCP tool."""
    async def _tool_fn(*args):
        try:
            return await mcp.function_wrapper(tool_name, *args)
        except Exception as e:
            # Wrap tool errors with context
            raise RuntimeError(f"Tool '{tool_name}' failed: {str(e)}") from e
    
    return _tool_fn
