"""
Custom LLM wrapper for GPT-5 models that use the responses.create() API.

GPT-5 models (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.2, gpt-5.2-pro) require the new
responses.create() API instead of chat.completions.create().
"""

from typing import Any, Dict, List, Optional, Iterator, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
import json
import time


# ============================================================================
# Research Depth to Model Parameters Mapping
# ============================================================================

def get_model_params_for_depth(
    model_name: str,
    research_depth: str,
    model_role: str = "quick"
) -> Dict[str, Any]:
    """
    Map research depth and model role to appropriate model parameters.
    
    Args:
        model_name: The LLM model name (e.g., "gpt-5-mini", "gpt-4o")
        research_depth: UI setting - "Shallow", "Medium", or "Deep"
        model_role: "quick" for Quick Thinker, "deep" for Deep Thinker
    
    Returns:
        Dict with appropriate model parameters (reasoning_effort, verbosity, temperature)
    
    Research Depth Strategy:
    
    | Depth    | Quick Thinker (speed-focused)  | Deep Thinker (quality-focused) |
    |----------|-------------------------------|-------------------------------|
    | Shallow  | effort: low, verbosity: low   | effort: medium, verbosity: medium |
    | Medium   | effort: medium, verbosity: medium | effort: high, verbosity: high |
    | Deep     | effort: high, verbosity: high | effort: xhigh, verbosity: high |
    """
    params = {}
    depth = research_depth.lower() if research_depth else "medium"
    role = model_role.lower() if model_role else "quick"
    
    # Check if it's a GPT-5 model
    if is_gpt5_model(model_name):
        # GPT-5 models use reasoning_effort and verbosity
        if role == "quick":
            # Quick thinker: prioritize speed, lower effort
            depth_mapping = {
                "shallow": {"reasoning_effort": "low", "verbosity": "low"},
                "medium": {"reasoning_effort": "medium", "verbosity": "medium"},
                "deep": {"reasoning_effort": "high", "verbosity": "high"},
            }
        else:  # deep thinker
            # Deep thinker: prioritize quality, higher effort
            depth_mapping = {
                "shallow": {"reasoning_effort": "medium", "verbosity": "medium"},
                "medium": {"reasoning_effort": "high", "verbosity": "high"},
                "deep": {"reasoning_effort": "xhigh", "verbosity": "high"},
            }
        
        mapping = depth_mapping.get(depth, depth_mapping["medium"])
        params["reasoning_effort"] = mapping["reasoning_effort"]
        params["verbosity"] = mapping["verbosity"]
        
    else:
        # Non-GPT-5 models (gpt-4o, gpt-4.1, o3, etc.) use temperature
        if role == "quick":
            # Quick thinker: more deterministic for speed
            temp_mapping = {
                "shallow": 0.1,
                "medium": 0.2,
                "deep": 0.3,
            }
        else:  # deep thinker
            # Deep thinker: slightly more creative for better analysis
            temp_mapping = {
                "shallow": 0.2,
                "medium": 0.3,
                "deep": 0.4,
            }
        
        # Only add temperature for models that support it
        no_temp_models = ["o3", "o3-mini", "o4-mini", "o1", "o1-mini", "o1-preview"]
        if not any(m in model_name for m in no_temp_models):
            params["temperature"] = temp_mapping.get(depth, 0.2)
    
    return params


def describe_model_params(
    model_name: str,
    research_depth: str,
    model_role: str = "quick"
) -> str:
    """Get a human-readable description of the model parameters being used."""
    params = get_model_params_for_depth(model_name, research_depth, model_role)
    
    if is_gpt5_model(model_name):
        return f"effort={params.get('reasoning_effort', 'medium')}, verbosity={params.get('verbosity', 'medium')}"
    else:
        temp = params.get('temperature')
        if temp:
            return f"temperature={temp}"
        return "default params"


class GPT5ChatModel(BaseChatModel):
    """Custom ChatModel wrapper for GPT-5 models using responses.create() API.
    
    Supported models and their parameters:
    
    gpt-5.2:
        - reasoning.effort: none, low, medium, high, xhigh
        - text.verbosity: low, medium, high
        - summary: concise, detailed, auto, null
    
    gpt-5.2-pro:
        - No reasoning effort
        - summary: concise, detailed, auto, null
    
    gpt-5:
        - reasoning.effort: minimal, low, medium, high
        - text.verbosity: low, medium, high
        - summary: concise, detailed, auto, null
    
    gpt-5.1:
        - reasoning.effort: none, low, medium, high
        - text.verbosity: low, medium, high
        - summary: concise, detailed, auto, null
    
    gpt-5-mini / gpt-5-nano:
        - reasoning.effort: minimal, low, medium, high
        - text.verbosity: low, medium, high
        - summary: concise, detailed, auto, null
    """
    
    model: str = "gpt-5-mini"
    api_key: Optional[str] = None
    reasoning_effort: str = "medium"  # Will be mapped to model-specific values
    verbosity: str = "medium"  # low, medium, high
    summary: str = "auto"  # concise, detailed, auto, null
    
    # Internal client - not a pydantic field
    _client: Optional[OpenAI] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the OpenAI client
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    def _get_model_type(self) -> str:
        """Determine the model type for parameter mapping."""
        if "gpt-5.2-pro" in self.model:
            return "gpt-5.2-pro"
        elif "gpt-5.2" in self.model:
            return "gpt-5.2"
        elif "gpt-5.1" in self.model:
            return "gpt-5.1"
        elif "gpt-5-mini" in self.model or "gpt-5-nano" in self.model:
            return "gpt-5-mini"  # Same params for mini and nano
        elif "gpt-5" in self.model:
            return "gpt-5"
        else:
            return "gpt-5-mini"  # Default
    
    def _map_reasoning_effort(self) -> Optional[str]:
        """Map reasoning effort to model-specific values."""
        model_type = self._get_model_type()
        effort = self.reasoning_effort.lower()
        
        # gpt-5.2-pro doesn't have reasoning effort
        if model_type == "gpt-5.2-pro":
            return None
        
        # Map common values to model-specific values
        effort_mapping = {
            "gpt-5.2": {
                # gpt-5.2 uses: none, low, medium, high, xhigh
                "minimal": "low",
                "none": "none",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "xhigh",
            },
            "gpt-5.1": {
                # gpt-5.1 uses: none, low, medium, high
                "minimal": "low",
                "none": "none",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
            },
            "gpt-5": {
                # gpt-5 uses: minimal, low, medium, high
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
            },
            "gpt-5-mini": {
                # gpt-5-mini/nano uses: minimal, low, medium, high
                "none": "minimal",
                "minimal": "minimal",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
            },
        }
        
        model_mapping = effort_mapping.get(model_type, effort_mapping["gpt-5-mini"])
        return model_mapping.get(effort, "medium")
    
    @property
    def _llm_type(self) -> str:
        return "gpt5-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "verbosity": self.verbosity,
        }
    
    def _convert_messages_to_input(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages, dicts, or strings to GPT-5 input format."""
        input_messages = []
        
        for message in messages:
            # Support dict-style messages used in some agents
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                # If content is already structured, pass through
                if isinstance(content, list):
                    input_messages.append({
                        "role": "developer" if role == "system" else role,
                        "content": content
                    })
                else:
                    content_type = "output_text" if role == "assistant" else "input_text"
                    input_messages.append({
                        "role": "developer" if role == "system" else role,
                        "content": [
                            {
                                "type": content_type,
                                "text": content
                            }
                        ]
                    })
                continue
            
            # Support raw string messages
            if isinstance(message, str):
                input_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message
                        }
                    ]
                })
                continue

            # Support generic message objects with role/content attributes
            if hasattr(message, "role") and hasattr(message, "content"):
                role = getattr(message, "role") or "user"
                content = getattr(message, "content", "")
                content_type = "output_text" if role == "assistant" else "input_text"
                input_messages.append({
                    "role": "developer" if role == "system" else role,
                    "content": [
                        {
                            "type": content_type,
                            "text": content
                        }
                    ]
                })
                continue

            if isinstance(message, SystemMessage):
                # GPT-5 uses "developer" role instead of "system"
                input_messages.append({
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message.content
                        }
                    ]
                })
            elif isinstance(message, HumanMessage):
                input_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message.content
                        }
                    ]
                })
            elif isinstance(message, AIMessage):
                input_messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": message.content
                        }
                    ]
                })
        
        return input_messages
    
    def _extract_content_from_response(self, response) -> tuple:
        """
        Extract text content and tool calls from GPT-5 response.
        Returns (content, tool_calls) tuple.
        
        Priority for content:
        1. response.output_text (convenience property with all text)
        2. Iterate through response.output items
        
        Tool calls are always extracted from response.output items.
        """
        tool_calls = []
        content_parts = []  # Use list to avoid duplication
        
        # Always iterate through output for tool calls
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                item_type = getattr(item, 'type', None)
                
                # Extract function calls
                if item_type == 'function_call':
                    call_id = getattr(item, 'call_id', None) or getattr(item, 'id', f'call_{len(tool_calls)+1}')
                    func_name = getattr(item, 'name', None)
                    func_args = getattr(item, 'arguments', {})
                    
                    if func_name:
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)
                            }
                        })
        
        # Get content - prefer output_text if available (it's the combined text)
        if hasattr(response, 'output_text') and response.output_text:
            return response.output_text.strip(), tool_calls
        
        # Fallback: extract text from output items
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                item_type = getattr(item, 'type', None)
                
                # Skip function calls and reasoning
                if item_type in ('function_call', 'reasoning'):
                    continue
                
                # Handle message content
                if item_type == 'message':
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            text = None
                            # Get text from content item
                            if hasattr(content_item, 'text') and content_item.text:
                                text = content_item.text
                            elif isinstance(content_item, dict):
                                text = content_item.get('text', '')
                            elif isinstance(content_item, str):
                                text = content_item
                            
                            if text and text not in content_parts:
                                content_parts.append(text)
                
                # Handle direct text output
                elif item_type in ('text', 'output_text'):
                    text = getattr(item, 'text', '')
                    if text and text not in content_parts:
                        content_parts.append(text)
        
        content = ''.join(content_parts)
        return content.strip(), tool_calls
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using GPT-5 responses.create() API."""
        
        # Convert messages to GPT-5 format
        input_messages = self._convert_messages_to_input(messages)
        if not input_messages and messages:
            # Fallback to prevent empty input errors
            input_messages = [{
                "role": "user",
                "content": [{"type": "input_text", "text": str(messages)}]
            }]
        
        # Determine model type and parameters
        model_type = self._get_model_type()
        reasoning_effort = self._map_reasoning_effort()
        
        # Build base API parameters
        api_params = {
            "model": self.model,
            "input": input_messages,
            "text": {
                "format": {"type": "text"}
            },
            "store": True,
        }
        
        # Add model-specific parameters based on model type
        if model_type == "gpt-5.2-pro":
            # gpt-5.2-pro: No reasoning effort, just summary
            if self.summary and self.summary != "null":
                api_params["summary"] = self.summary
                
        elif model_type == "gpt-5.2":
            # gpt-5.2: reasoning.effort (none/low/medium/high/xhigh), verbosity, summary
            if reasoning_effort:
                api_params["reasoning"] = {"effort": reasoning_effort}
            else:
                 api_params["reasoning"] = {}

            api_params["text"]["verbosity"] = self.verbosity
            if self.summary and self.summary != "null":
                api_params["reasoning"]["summary"] = self.summary
                
        elif model_type == "gpt-5.1":
            # gpt-5.1: reasoning.effort (none/low/medium/high), verbosity, summary
            if reasoning_effort:
                api_params["reasoning"] = {"effort": reasoning_effort}
            api_params["text"]["verbosity"] = self.verbosity
            if self.summary and self.summary != "null":
                api_params["reasoning"]["summary"] = self.summary
                
        else:
            # gpt-5, gpt-5-mini, gpt-5-nano: reasoning.effort (minimal/low/medium/high), verbosity, summary
            api_params["reasoning"] = {}
            if reasoning_effort:
                api_params["reasoning"]["effort"] = reasoning_effort
            if self.summary and self.summary != "null":
                api_params["reasoning"]["summary"] = self.summary
            api_params["text"]["verbosity"] = self.verbosity
        
        print(f"[GPT5] Model: {self.model} (type: {model_type}), effort: {reasoning_effort}, verbosity: {self.verbosity}")
        input_chars = len(str(input_messages))

        def _extract_usage_dict(resp) -> Dict[str, int]:
            usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            usage = getattr(resp, "usage", None)
            if usage is None:
                return usage_dict

            if isinstance(usage, dict):
                usage_dict["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
                usage_dict["output_tokens"] = int(usage.get("output_tokens", 0) or 0)
                usage_dict["total_tokens"] = int(
                    usage.get("total_tokens", usage_dict["input_tokens"] + usage_dict["output_tokens"]) or 0
                )
                return usage_dict

            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = getattr(usage, key, None)
                if value is not None:
                    usage_dict[key] = int(value or 0)

            if usage_dict["total_tokens"] == 0:
                usage_dict["total_tokens"] = usage_dict["input_tokens"] + usage_dict["output_tokens"]
            return usage_dict

        def _record_llm_call(
            *,
            status: str,
            latency_seconds: float,
            output_chars: int,
            usage: Dict[str, int] | None = None,
            error_message: str | None = None,
        ) -> None:
            payload = {
                "model": self.model,
                "purpose": "gpt5_responses",
                "status": status,
                "latency_seconds": round(float(latency_seconds), 4),
                "input_chars": input_chars,
                "output_chars": int(output_chars or 0),
                "effort": reasoning_effort,
                "verbosity": self.verbosity,
                "usage": usage or {},
                "error_message": error_message,
            }

            logged_to_state = False
            try:
                from webui.utils.state import app_state

                app_state.register_llm_call(
                    model_name=self.model,
                    purpose="gpt5_responses",
                    latency_seconds=payload["latency_seconds"],
                    input_chars=payload["input_chars"],
                    output_chars=payload["output_chars"],
                    effort=payload["effort"],
                    verbosity=payload["verbosity"],
                    usage=payload["usage"],
                    status=status,
                    error_message=error_message,
                )
                logged_to_state = True
            except Exception:
                pass

            if not logged_to_state:
                try:
                    from tradingagents.run_logger import get_run_audit_logger
                    get_run_audit_logger().log_event(event_type="llm_call", payload=payload)
                except Exception:
                    pass
        
        # Handle tool calls if present
        tools = kwargs.get("tools", [])
        if tools:
            # Convert LangChain tools to OpenAI function format for GPT-5
            openai_tools = []
            for tool in tools:
                tool_name = None
                tool_description = None
                tool_parameters = None
                
                # Try to get tool info from different possible attributes
                if hasattr(tool, 'name'):
                    tool_name = tool.name
                elif hasattr(tool, 'func') and hasattr(tool.func, '__name__'):
                    tool_name = tool.func.__name__
                
                if hasattr(tool, 'description'):
                    tool_description = tool.description
                elif hasattr(tool, 'func') and hasattr(tool.func, '__doc__'):
                    tool_description = tool.func.__doc__ or "No description"
                
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        tool_parameters = tool.args_schema.schema()
                    except Exception:
                        tool_parameters = {"type": "object", "properties": {}}
                else:
                    tool_parameters = {"type": "object", "properties": {}}
                
                # Only add tools that have a valid name
                if tool_name:
                    tool_schema = {
                        "type": "function",
                        "name": tool_name,  # GPT-5 expects name at top level
                        "description": tool_description or f"Tool: {tool_name}",
                        "parameters": tool_parameters
                    }
                    openai_tools.append(tool_schema)
            
            if openai_tools:
                api_params["tools"] = openai_tools
        
        try:
            # Make the API call
            call_started = time.time()
            response = self._client.responses.create(**api_params)
            latency_seconds = time.time() - call_started
            
            # Debug: Print response structure
            if hasattr(response, 'output') and response.output:
                print(f"[GPT5] Response has {len(response.output)} output items")
                for i, item in enumerate(response.output):
                    item_type = getattr(item, 'type', 'unknown')
                    print(f"[GPT5]   Item {i}: type={item_type}")
                    if item_type == 'function_call':
                        print(f"[GPT5]     Function: {getattr(item, 'name', 'unknown')}")
                    elif item_type == 'message':
                        # Debug: show message structure
                        if hasattr(item, 'content'):
                            print(f"[GPT5]     Message has content: {type(item.content)}")
                            if isinstance(item.content, list):
                                for j, c in enumerate(item.content):
                                    c_type = getattr(c, 'type', type(c).__name__)
                                    c_text = getattr(c, 'text', str(c)[:100] if c else 'None')
                                    print(f"[GPT5]       Content[{j}]: type={c_type}, text_len={len(c_text) if c_text else 0}")
                            elif isinstance(item.content, str):
                                print(f"[GPT5]       Content is string, len={len(item.content)}")
                        else:
                            print(f"[GPT5]     Message has no content attr, dir: {[a for a in dir(item) if not a.startswith('_')]}")
            
            # Extract content and tool calls
            content, tool_calls = self._extract_content_from_response(response)
            
            # Debug: Show extracted content
            if content:
                print(f"[GPT5] Extracted content: {len(content)} chars")
                print(f"[GPT5]   Content preview: {content[:200]}..." if len(content) > 200 else f"[GPT5]   Content: {content}")
            else:
                print(f"[GPT5] WARNING: No content extracted from response!")
                # Try to get output_text directly
                if hasattr(response, 'output_text') and response.output_text:
                    print(f"[GPT5]   But output_text exists: {len(response.output_text)} chars")
                    content = response.output_text
            
            if tool_calls:
                print(f"[GPT5] Found {len(tool_calls)} tool calls")
                for tc in tool_calls:
                    print(f"[GPT5]   Tool: {tc['function']['name']}")

            usage_dict = _extract_usage_dict(response)
            _record_llm_call(
                status="success",
                latency_seconds=latency_seconds,
                output_chars=len(content or ""),
                usage=usage_dict,
            )
            
            # Create the AI message
            additional_kwargs = {}
            if tool_calls:
                additional_kwargs["tool_calls"] = tool_calls
            
            # Also add tool_calls attribute for LangChain compatibility
            ai_message = AIMessage(
                content=content,
                additional_kwargs=additional_kwargs
            )
            
            # Set tool_calls attribute directly for better compatibility
            if tool_calls:
                ai_message.tool_calls = [
                    {
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                        "id": tc["id"],
                        "type": "tool_call"
                    }
                    for tc in tool_calls
                ]
            
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            # Return error as content
            error_message = f"Error calling GPT-5 API: {str(e)}"
            print(f"[GPT5] {error_message}")
            latency_seconds = 0.0
            try:
                latency_seconds = max(0.0, time.time() - call_started)
            except Exception:
                pass
            _record_llm_call(
                status="error",
                latency_seconds=latency_seconds,
                output_chars=len(error_message),
                usage={},
                error_message=error_message,
            )
            ai_message = AIMessage(content=error_message)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
    
    def bind_tools(self, tools: List[Any], **kwargs) -> "GPT5ChatModel":
        """Bind tools to the model for function calling."""
        # Create a new instance with tools stored
        new_model = GPT5ChatModel(
            model=self.model,
            api_key=self.api_key,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
            summary=self.summary,
        )
        new_model._bound_tools = tools
        return new_model
    
    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> AIMessage:
        """Invoke the model with input."""
        # Handle string input
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        elif isinstance(input, list):
            messages = input
        else:
            messages = [HumanMessage(content=str(input))]
        
        # Add bound tools if present
        if hasattr(self, '_bound_tools'):
            kwargs['tools'] = self._bound_tools
        
        result = self._generate(messages, **kwargs)
        return result.generations[0].message


def is_gpt5_model(model_name: str) -> bool:
    """Check if a model name is a GPT-5 variant."""
    gpt5_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2", "gpt-5.2-pro"]
    return any(model_prefix in model_name for model_prefix in gpt5_models)


def get_chat_model(model_name: str, api_key: Optional[str] = None, **kwargs):
    """
    Factory function to get the appropriate chat model based on model name.
    
    Returns GPT5ChatModel for GPT-5 variants, otherwise returns ChatOpenAI.
    
    Supported GPT-5 models:
    - gpt-5, gpt-5-mini, gpt-5-nano: reasoning.effort (minimal/low/medium/high)
    - gpt-5.1: reasoning.effort (none/low/medium/high)
    - gpt-5.2: reasoning.effort (none/low/medium/high/xhigh)
    - gpt-5.2-pro: No reasoning effort, just summary
    """
    if is_gpt5_model(model_name):
        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
        verbosity = kwargs.pop("verbosity", "medium")
        summary = kwargs.pop("summary", "auto")
        
        # Remove temperature if present (GPT-5 doesn't support it)
        kwargs.pop("temperature", None)
        
        return GPT5ChatModel(
            model=model_name,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            summary=summary,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, openai_api_key=api_key, **kwargs)
