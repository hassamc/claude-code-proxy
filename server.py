from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import openai
import asyncio

# Import Rich for better debugging output
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table, Column
from rich import print as rprint
from rich.console import Group

# Create a rich console for debugging
debug_console = RichConsole()

# Flag to enable rich debugging independently of log level
RICH_DEBUG = True # Set to True to enable rich debugging by default

# Flag to control newline handling in Together.ai responses
PRESERVE_TOGETHER_FORMATTING = False # Set to False to process newlines for Together.ai
# Flag to control OpenAI client newline handling
PRESERVE_OPENAI_NEWLINES = False # Set to False to keep \n as \n in OpenAI responses

# Load environment variables from .env file
load_dotenv()

# Configure logging based on environment variable
log_level_str = os.environ.get("LOGLEVEL", "WARN").upper()
log_level = getattr(logging, log_level_str, logging.WARN)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
logger.info(f"Logger initialized with level: {log_level_str}")

# Check if we should show detailed model information
SHOW_MODEL_DETAILS = os.environ.get("SHOW_MODEL_DETAILS", "").lower() == "true"
if SHOW_MODEL_DETAILS:
    logger.info("Model details logging enabled")

# Enable LiteLLM debugging only in full debug mode
if log_level == logging.DEBUG:
    import litellm
    litellm._turn_on_debug()
    logger.info("LiteLLM debug mode enabled")

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TOGETHERAI_API_KEY = os.environ.get("TOGETHERAI_API_KEY")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# Default Together.ai model
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash"
]

# List of Together.ai models
TOGETHER_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
]

# Helper function to ensure model name is compatible with Together.ai
def ensure_together_model_format(model_name):
    """Ensures model name is properly formatted for Together.ai API"""
    # If model doesn't include organization prefix, add it
    if "/" not in model_name and model_name not in ["llama-3-70b-chat", "llama-3-8b-chat"]:
        logger.warning(f"Model {model_name} missing organization prefix, using with caution")
    
    # Strip together_ai/ prefix if it exists
    if model_name.startswith("together_ai/"):
        model_name = model_name[12:]
    
    return model_name

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True
    type: Optional[str] = None
    budget_tokens: Optional[int] = None

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('together_ai/'):
            clean_v = clean_v[12:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "together":
                new_model = f"together_ai/{TOGETHER_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "together":
                new_model = f"together_ai/{TOGETHER_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in TOGETHER_MODELS and not v.startswith('together_ai/'):
                new_model = f"together_ai/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'together_ai/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('together_ai/'):
            clean_v = clean_v[12:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "together":
                new_model = f"together_ai/{TOGETHER_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "together":
                new_model = f"together_ai/{TOGETHER_MODEL}"
                mapped = True
            elif PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in TOGETHER_MODELS and not v.startswith('together_ai/'):
                new_model = f"together_ai/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'together_ai/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def detect_tool_calls_in_text(text_content):
    """
    Parse text output from Together.ai models to detect intended function calls.
    Returns a list of detected tool calls in a format compatible with OpenAI's tool_calls.
    """
    try:
        tool_calls = []
        
        # Pattern 1: Look for markdown-style code blocks with JSON
        # Example: ```json\n{"tool": "weather", "params": {"location": "New York"}}\n```
        code_block_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        code_blocks = re.findall(code_block_pattern, text_content, re.DOTALL)
        
        # Pattern 2: Look for explicit function call syntax
        # Example: USE_TOOL: calculator PARAMETERS: {"x": 5, "y": 10, "operation": "multiply"}
        function_pattern = r"USE_TOOL:\s*(\w+)\s*PARAMETERS:\s*(\{.*?\})"
        function_matches = re.findall(function_pattern, text_content, re.DOTALL)
        
        # Pattern 3: Look for natural language function references with JSON-like content
        # Example: I'll use the weather tool with parameters: {"location": "New York"}
        nl_pattern = r"use the (\w+) tool with parameters:?\s*(\{.*?\})"
        nl_matches = re.findall(nl_pattern, text_content, re.IGNORECASE | re.DOTALL)
        
        # Process code blocks as potential tool calls
        for block in code_blocks:
            try:
                # Try to parse as JSON
                tool_data = json.loads(block.strip())
                if isinstance(tool_data, dict) and "tool" in tool_data and "params" in tool_data:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "function": {
                            "name": tool_data["tool"],
                            "arguments": json.dumps(tool_data["params"])
                        }
                    })
            except json.JSONDecodeError:
                # Not valid JSON, skip
                continue
        
        # Process explicit function calls
        for name, args_str in function_matches:
            try:
                # Try to parse arguments as JSON
                args = json.loads(args_str.strip())
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args)
                    }
                })
            except json.JSONDecodeError:
                # Not valid JSON, try to fix common issues and retry
                fixed_args_str = fix_json_string(args_str)
                try:
                    args = json.loads(fixed_args_str)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args)
                        }
                    })
                except json.JSONDecodeError:
                    # Still not valid, skip
                    continue
        
        # Process natural language matches
        for name, args_str in nl_matches:
            try:
                # Try to parse arguments as JSON
                args = json.loads(args_str.strip())
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args)
                    }
                })
            except json.JSONDecodeError:
                # Not valid JSON, try to fix common issues and retry
                fixed_args_str = fix_json_string(args_str)
                try:
                    args = json.loads(fixed_args_str)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args)
                        }
                    })
                except json.JSONDecodeError:
                    # Still not valid, skip
                    continue
        
        return tool_calls
    except Exception as e:
        logger.error(f"Error detecting tool calls: {str(e)}")
        # Log the problematic text for debugging
        logger.debug(f"Text content that caused error: {text_content[:500]}...")
        return []  # Return empty list on error

def fix_json_string(json_str):
    """Attempt to fix common JSON formatting issues"""
    # Replace single quotes with double quotes
    fixed = re.sub(r"'([^']*)'", r'"\1"', json_str)
    # Ensure property names are properly quoted
    fixed = re.sub(r'(\w+):', r'"\1":', fixed)
    # Remove trailing commas
    fixed = re.sub(r',\s*}', '}', fixed)
    return fixed

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-x" format
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = parse_tool_result_content(block.content)
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            processed_content_block["content"] = parse_tool_result_content(block.content)
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Use client-provided max_tokens value by default
    max_tokens = anthropic_request.max_tokens
    
    # For logging only - don't modify the value
    if SHOW_MODEL_DETAILS:
        logger.info(f"Client requested max_tokens: {max_tokens}")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "stream": anthropic_request.stream,
    }
    
    # Special handling for o3 and o4 models
    if anthropic_request.model.startswith("openai/o3") or anthropic_request.model.startswith("openai/o4"):
        # OpenAI o3/o4 models use max_completion_tokens instead of max_tokens
        # For Claude Code, we should fully respect the client's token request
        litellm_request["max_completion_tokens"] = max_tokens
        
        if SHOW_MODEL_DETAILS:
            logger.info(f"Using max_completion_tokens={max_tokens} for {anthropic_request.model}")
        
        # Override system prompt for o3 models to experiment with its behavior
        if anthropic_request.model.startswith("openai/o3"):
            # Log the original system message if any
            original_system = None
            system_msg_idx = -1
            
            # Find any existing system message
            for idx, msg in enumerate(litellm_request["messages"]):
                if msg.get("role") == "system":
                    system_msg_idx = idx
                    original_system = msg.get("content")
                    if original_system:
                        # Print the full system prompt split into reasonably sized chunks for logging
                        logger.info(f"📥 ORIGINAL SYSTEM PROMPT (FULL):")
                        # Split into chunks of around 1000 chars for readability in logs
                        chunk_size = 1000
                        for i in range(0, len(original_system), chunk_size):
                            chunk = original_system[i:i+chunk_size]
                            logger.info(f"SYSTEM PROMPT PART {i//chunk_size + 1}: {chunk}")
                    break
            
            # Instead of replacing, let's modify the existing system prompt by injecting personality guidelines
            if original_system:
                # Locate the tone instruction about being concise
                concise_pattern = r"You should be concise, direct, and to the point"
                
                replacement = "You should be concise, direct, to the point, and also friendly and a good coworker"
                
                modified_system_prompt = re.sub(concise_pattern, replacement, original_system)
                
                # Remove overly strict brevity constraints
                brevity_patterns = [
                    r"You MUST answer concisely with fewer than 4 lines[^\n]*\n?",
                    r"One word answers are best\.?:?[^\n]*\n?"
                ]
                for pat in brevity_patterns:
                    modified_system_prompt = re.sub(pat, "", modified_system_prompt, flags=re.IGNORECASE)
                
                # Inject softer, friendlier guidance
                personality_addendum = """\n\nADDITIONAL GUIDANCE:\n- Use a warm, conversational tone while remaining professional.\n- Feel free to elaborate when helpful; you are not limited to four lines.\n- Aim to be supportive, encouraging, and collaborative."""
                
                if "ADDITIONAL GUIDANCE:" not in modified_system_prompt:
                    modified_system_prompt += personality_addendum
                
                # Check if we actually made a change
                if modified_system_prompt != original_system:
                    logger.info(f"🔄 Modified system prompt: Added friendliness and personality guidance")
                    
                    # Update the system message with our modified version
                    litellm_request["messages"][system_msg_idx]["content"] = modified_system_prompt
                else:
                    logger.info(f"⚠️ Could not modify system prompt - pattern not found")
            else:
                # No system prompt found, create a simple one
                simple_prompt = "You are Claude, an AI assistant. You should be concise, direct, to the point, and also friendly and a good coworker. Provide useful, informative responses."
                litellm_request["messages"].insert(0, {"role": "system", "content": simple_prompt})
                logger.info(f"➕ Added a simple system prompt (no original found)")
                
            # Log all messages being sent (truncated for readability)
            for i, msg in enumerate(litellm_request["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"📩 MESSAGE {i}: role={role}, content={content_preview}")
        
        # o3 and o4 models don't support custom temperature - they only support the default (1.0)
        # Only add temperature if it's the default value of 1.0
        if anthropic_request.temperature == 1.0:
            litellm_request["temperature"] = 1.0
    else:
        # For other models, use standard parameters
        litellm_request["max_tokens"] = max_tokens
        litellm_request["temperature"] = anthropic_request.temperature
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Check if this is a Together.ai model
    is_together_model = anthropic_request.model.startswith("together_ai/")
    
    # Add rich debug logging for Together.ai models
    if (is_together_model and log_level <= logging.DEBUG) or RICH_DEBUG:
        debug_console.print(Panel(
            f"[bold blue]Converting Anthropic request for Together.ai[/bold blue]\n"
            f"Model: [cyan]{anthropic_request.model}[/cyan]\n"
            f"Tools present: [{'yellow' if anthropic_request.tools else 'green'}]{bool(anthropic_request.tools)}[/{'yellow' if anthropic_request.tools else 'green'}]\n"
            f"Message count: {len(anthropic_request.messages)}\n"
            f"Max tokens: {anthropic_request.max_tokens}",
            title="[bold]Together.ai Request Conversion[/bold]",
            border_style="blue"
        ))
    
    # Convert tools to OpenAI format for ALL models including Together.ai
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
        # Convert tool_choice to OpenAI format if present
        if anthropic_request.tool_choice:
            if hasattr(anthropic_request.tool_choice, 'dict'):
                tool_choice_dict = anthropic_request.tool_choice.dict()
            else:
                tool_choice_dict = anthropic_request.tool_choice
                
            # Handle Anthropic's tool_choice format
            choice_type = tool_choice_dict.get("type")
            if choice_type == "auto":
                litellm_request["tool_choice"] = "auto"
            elif choice_type == "any":
                litellm_request["tool_choice"] = "auto"
                logger.info("Mapping tool_choice 'any' to 'auto' for Together.ai")
            elif choice_type == "tool" and "name" in tool_choice_dict:
                litellm_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice_dict["name"]}
                }
            else:
                # Default to auto if we can't determine
                litellm_request["tool_choice"] = "auto"
        else:
            # Set default tool_choice to auto if tools are provided but no explicit choice
            litellm_request["tool_choice"] = "auto"
            
        # Special handling for Together.ai with tools - enhanced system prompt
        if is_together_model:
            # Get the original system prompt, or create a new one
            system_prompt = ""
            system_msg_idx = -1
            
            for idx, msg in enumerate(litellm_request["messages"]):
                if msg.get("role") == "system":
                    system_msg_idx = idx
                    system_prompt = msg.get("content", "")
                    break
            
            # Add specific Together.ai function calling instructions to help guide the model
            function_calling_instructions = """
When you use tools/functions, ensure you follow these guidelines:
1. Call one function at a time
2. Wait for the result before calling another function
3. Format function calls using the API's native function calling capability
4. Do not format function calls as text or code blocks
"""
            
            # Update the system prompt
            if system_msg_idx >= 0:
                litellm_request["messages"][system_msg_idx]["content"] = system_prompt + "\n\n" + function_calling_instructions
            else:
                litellm_request["messages"].insert(0, {"role": "system", "content": "You are a helpful assistant.\n\n" + function_calling_instructions})
                
            # Log what we're doing
            logger.info(f"Using native function calling for Together.ai model with {len(openai_tools)} tools")
    
    # Special handling for Together.ai models
    if is_together_model:
        # Update the model name for Together.ai
        # Strip the provider prefix
        if litellm_request["model"].startswith("together_ai/"):
            litellm_request["model"] = litellm_request["model"][len("together_ai/"):]
        
        # Set provider explicitly to together_ai
        litellm_request["custom_llm_provider"] = "together_ai"
        
        # Set the base URL for Together API
        litellm_request["base_url"] = "https://api.together.xyz/v1"
        
        # Set API key
        litellm_request["api_key"] = TOGETHERAI_API_KEY
        
        if log_level <= logging.DEBUG or RICH_DEBUG:
            # Create a sanitized version of the request for debug output
            debug_request = litellm_request.copy()
            if "api_key" in debug_request:
                api_key = debug_request["api_key"]
                if api_key:
                    debug_request["api_key"] = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***masked***"
            
            # Show details about the configured tools
            tools_info = ""
            if "tools" in litellm_request:
                tool_count = len(litellm_request["tools"])
                tool_names = [t["function"]["name"] for t in litellm_request["tools"]]
                tools_info = f"\nTools ({tool_count}): {', '.join(tool_names)}\n"
                tools_info += f"Tool choice: {litellm_request.get('tool_choice', 'auto')}"
            
            debug_console.print(Panel(
                f"[bold green]Together.ai Native Function Calling[/bold green]\n"
                f"Model: [cyan]{litellm_request['model']}[/cyan]\n"
                f"API Key: {debug_request.get('api_key', 'Not set')}\n"
                f"Provider: {litellm_request.get('custom_llm_provider')}\n"
                f"Base URL: {litellm_request.get('base_url')}{tools_info}",
                title="[bold]Together.ai Configuration[/bold]",
                border_style="green"
            ))
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        elif clean_model.startswith("together_ai/"):
            clean_model = clean_model[len("together_ai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Process content text for Together.ai
        # Together.ai might return newlines encoded as literal \n that need proper handling
        if original_request.model.startswith("together_ai/") and isinstance(content_text, str):
            # Get provider name for debug messages
            provider_name = "Together.ai"
            
            # Special handling for Together.ai models to detect tool calls in text content
            # First check if there were tools in the original request
            has_tools = original_request.tools and len(original_request.tools) > 0
            
            if has_tools:
                # Detect tool calls in the model's text output
                detected_tool_calls = detect_tool_calls_in_text(content_text)
                
                if detected_tool_calls:
                    # Log that we detected tool calls
                    if log_level <= logging.DEBUG or RICH_DEBUG:
                        debug_console.print(Panel(
                            f"[bold green]{provider_name} tool calls detected[/bold green]\n"
                            f"Found {len(detected_tool_calls)} tool call(s) in text response",
                            title="[bold]Tool Call Detection[/bold]",
                            border_style="green"
                        ))
                        
                        # Show the detected tool calls
                        for i, call in enumerate(detected_tool_calls):
                            debug_console.print(f"[bold]Tool Call {i+1}:[/bold]")
                            debug_console.print(f"Name: {call['function']['name']}")
                            debug_console.print(f"Arguments: {call['function']['arguments']}")
                    
                    # If we're using a Claude model as the output format, add proper tool_use blocks
                    if is_claude_model:
                        # Add detected tool calls as tool_use blocks
                        for tool_call in detected_tool_calls:
                            # Extract function data
                            tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                            name = tool_call['function']['name']
                            arguments_str = tool_call['function']['arguments']
                            
                            # Convert arguments string to dict
                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                arguments = {"raw": arguments_str}
                            
                            # Add tool_use block
                            content.append({
                                "type": "tool_use",
                                "id": tool_id,
                                "name": name,
                                "input": arguments
                            })
                    
                        # Extract text content without the tool calls to avoid duplication
                        if detected_tool_calls and content and content[0]["type"] == "text":
                            # Remove the tool call parts from the text content
                            cleaned_text = content_text
                            
                            # Remove code blocks with JSON
                            cleaned_text = re.sub(r"```(?:json)?\s*\n.*?\n```", "[Tool Call Removed]", cleaned_text, flags=re.DOTALL)
                            
                            # Remove explicit function calls
                            cleaned_text = re.sub(r"USE_TOOL:\s*\w+\s*PARAMETERS:\s*\{.*?\}", "[Tool Call Removed]", cleaned_text, flags=re.DOTALL)
                            
                            # Remove natural language function references
                            cleaned_text = re.sub(r"use the \w+ tool with parameters:?\s*\{.*?\}", "[Tool Call Removed]", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
                            
                            # Clean up any artifact double-spaces or empty lines
                            cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
                            cleaned_text = re.sub(r"\[Tool Call Removed\]\s*\[Tool Call Removed\]", "[Tool Call Removed]", cleaned_text)
                            
                            # Update the text content
                            content[0]["text"] = cleaned_text.strip()
                    
                    # Set stop reason to "tool_use" if we detected tool calls
                    if detected_tool_calls:
                        stop_reason = "tool_use"
        
            # Only process newlines if PRESERVE_TOGETHER_FORMATTING is False
            if not PRESERVE_TOGETHER_FORMATTING:
                # Replace escaped newlines with actual newlines
                if "\\n" in content_text:
                    content_text = content_text.replace("\\n", "\n")
                
                # Fix markdown code blocks that might be malformed
                if "```" in content_text:
                    # Ensure code blocks have proper newlines
                    content_text = re.sub(r'```(\w+)([^`]+)```', r'```\1\n\2\n```', content_text)
                
                if log_level <= logging.DEBUG or RICH_DEBUG:
                    debug_console.print(Panel(
                        f"[bold yellow]{provider_name} response transformed[/bold yellow]\n"
                        f"Processing newlines for readability",
                        title="[bold]Response Handling[/bold]",
                        border_style="yellow"
                    ))
            else:
                # Debug logging for content when preserving original formatting
                if log_level <= logging.DEBUG or RICH_DEBUG:
                    debug_console.print(Panel(
                        f"[bold blue]{provider_name} response untransformed[/bold blue]\n"
                        f"Preserving original formatting for Anthropic client",
                        title="[bold]Response Handling[/bold]",
                        border_style="cyan"
                    ))
        
        # Process content text for OpenAI clients
        # OpenAI clients may need to convert actual newlines to literal \n for proper display
        elif not original_request.model.startswith("together_ai/") and isinstance(content_text, str) and not PRESERVE_OPENAI_NEWLINES:
            # Get provider name for debug messages
            provider_name = "OpenAI"
            if original_request.model.startswith("openai/"):
                provider_name = "OpenAI"
            elif original_request.model.startswith("gemini/"):
                provider_name = "Gemini"
            elif original_request.model.startswith("anthropic/"):
                provider_name = "Anthropic"
            
            # Replace actual newlines with literal \n for OpenAI output if needed
            if "\n" in content_text:
                content_text = content_text.replace("\n", "\\n")
                
                if log_level <= logging.DEBUG or RICH_DEBUG:
                    debug_console.print(Panel(
                        f"[bold green]{provider_name} response transformed[/bold green]\n"
                        f"Converting actual newlines to \\n literals for OpenAI client display",
                        title="[bold]Response Handling[/bold]",
                        border_style="green"
                    ))
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - for all models including Together.ai
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """
    Process the streaming response from LiteLLM and convert it to Anthropic's streaming format.
    """
    try:
        # Initialize variables to track state across chunks
        content_blocks = {}  # Store content blocks by index
        tool_index = None  # Current tool index being processed
        last_tool_index = -1  # Highest tool index seen
        has_sent_stop_reason = False  # Whether we've sent a stop reason
        output_tokens = 0  # Count tokens for final usage info
        
        # Special tracking for Together.ai with tools
        is_together_model = original_request.model.startswith("together_ai/")
        has_tools = original_request.tools and len(original_request.tools) > 0
        accumulated_text = "" if (is_together_model and has_tools) else None
        detected_tool_calls = []
        text_sent = False
        text_block_closed = False

        # Process each chunk in the streaming response
        async for chunk in response_generator:
            # Count output tokens
            delta = chunk.choices[0].delta if hasattr(chunk.choices[0], 'delta') else {}
            if hasattr(delta, 'content') and delta.content:
                output_tokens += 1  # Simple approximation
                
                # For Together.ai with tools, accumulate text
                if is_together_model and has_tools:
                    accumulated_text += delta.content

            # Extract message data from the chunk
            message = chunk.choices[0].delta if hasattr(chunk.choices[0], 'delta') else None
            content = message.content if message and hasattr(message, 'content') else None
            
            # Handle tool calls
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            
            # Check if this is the start of the response
            if content and tool_index is None:
                # Start with text content at index 0
                tool_index = 0
                text_sent = True
                
                # Send message_start event
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start'})}\n\n"
                
                # Start a text content block
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text'}})}\n\n"
                
                # Set up the first content block
                content_blocks[0] = {"type": "text", "text": ""}
            
            # Process text content
            if content:
                # Ensure we have a text block
                if 0 not in content_blocks:
                    # First text - start a message and content block
                    tool_index = 0
                    text_sent = True
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start'})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text'}})}\n\n"
                    content_blocks[0] = {"type": "text", "text": ""}
                
                # Add to the text content
                content_blocks[0]["text"] += content
                
                # Send content delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                
                # Check for Together.ai tool calls in accumulated text
                if is_together_model and has_tools and accumulated_text and not detected_tool_calls:
                    # Check for complete tool call patterns
                    # We need to be careful about detecting complete patterns, not partial ones
                    complete_json_block = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", accumulated_text, re.DOTALL)
                    complete_function_call = re.search(r"USE_TOOL:\s*(\w+)\s*PARAMETERS:\s*(\{.*?\})", accumulated_text, re.DOTALL)
                    
                    if complete_json_block or complete_function_call:
                        # We've found a complete tool call pattern
                        tool_calls_found = detect_tool_calls_in_text(accumulated_text)
                        
                        if tool_calls_found:
                            # First time we've found tool calls
                            detected_tool_calls = tool_calls_found
                            
                            # Close the text block if it was active
                            if not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                                # Create a cleaned version of text content without the tool calls
                                cleaned_text = accumulated_text
                                cleaned_text = re.sub(r"```(?:json)?\s*\n.*?\n```", "[Tool Call Removed]", cleaned_text, flags=re.DOTALL)
                                cleaned_text = re.sub(r"USE_TOOL:\s*\w+\s*PARAMETERS:\s*\{.*?\}", "[Tool Call Removed]", cleaned_text, flags=re.DOTALL)
                                cleaned_text = re.sub(r"use the \w+ tool with parameters:?\s*\{.*?\}", "[Tool Call Removed]", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
                                cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
                                
                                # Replace content_blocks[0] with cleaned text
                                content_blocks[0]["text"] = cleaned_text.strip()
                            
                            # Start tool_use blocks for each detected tool call
                            for i, tool_call in enumerate(detected_tool_calls):
                                tool_index = i + 1
                                last_tool_index = max(last_tool_index, tool_index)
                                
                                # Extract function info
                                function = tool_call.get('function', {})
                                name = function.get('name', '')
                                tool_id = tool_call.get('id', f"tool_{uuid.uuid4().hex[:8]}")
                                
                                # Start a new tool_use block
                                content_blocks[tool_index] = {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": name,
                                    "input": {}
                                }
                                
                                content_block_data = {
                                    'type': 'content_block_start',
                                    'index': tool_index,
                                    'content_block': {
                                        'type': 'tool_use',
                                        'id': tool_id,
                                        'name': name,
                                        'input': {}
                                    }
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(content_block_data)}\n\n"
                                
                                # Add the tool arguments
                                arguments_str = function.get('arguments', '{}')
                                try:
                                    arguments = json.loads(arguments_str)
                                    content_blocks[tool_index]["input"] = arguments
                                    
                                    # Send input delta
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tool_index, 'delta': {'type': 'tool_use_delta', 'input': arguments}})}\n\n"
                                    
                                    # Close the tool_use block
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"
                                except json.JSONDecodeError:
                                    # Send as string if not valid JSON
                                    content_blocks[tool_index]["input"] = {"raw": arguments_str}
                                    
                                    # Send input delta
                                    raw_arg_object = {'raw': arguments_str}
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tool_index, 'delta': {'type': 'tool_use_delta', 'input': raw_arg_object}})}\n\n"
                                    
                                    # Close the tool_use block
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"
                            
                            # Set stop reason to tool_use since we detected tool calls
                            stop_reason = "tool_use"
                            
                            # Send message_delta with stop reason
                            usage = {"output_tokens": output_tokens}
                            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': usage})}\n\n"
                            
                            # Set flag to avoid duplicate stop messages
                            has_sent_stop_reason = True
                            
                            # Send message_stop event
                            message_stop_json = json.dumps({'type': 'message_stop'})
                            yield f"event: message_stop\ndata: {message_stop_json}\n\n"
                            
                            # Send final [DONE] marker
                            yield "data: [DONE]\n\n"
                            return
            
            # Process tool calls from OpenAI API directly
            if tool_calls:
                for tool_call in tool_calls:
                    # Only process if there's a function
                    if not hasattr(tool_call, 'function'):
                        continue
                    
                    # Determine tool index (starting at 1 as 0 is reserved for text)
                    if not hasattr(tool_call, 'index'):
                        # If no index, increment based on our own last seen
                        last_tool_index += 1
                        current_tool_index = last_tool_index
                    else:
                        # Use the provided index + 1 (since 0 is text)
                        current_tool_index = tool_call.index + 1
                        last_tool_index = max(last_tool_index, current_tool_index)
                    
                    # Initialize the tool block if not present
                    if current_tool_index not in content_blocks:
                        tool_name = ""
                        if hasattr(tool_call.function, 'name'):
                            tool_name = tool_call.function.name
                        
                        # Generate a unique ID for this tool call
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"
                        
                        # Start a new tool use block
                        content_blocks[current_tool_index] = {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": {}
                        }
                        
                        # Send tool block start event
                        content_block_data = {
                            'type': 'content_block_start',
                            'index': current_tool_index,
                            'content_block': {
                                'type': 'tool_use',
                                'id': tool_id,
                                'name': tool_name,
                                'input': {}
                            }
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(content_block_data)}\n\n"
                    
                    # Update tool name if provided
                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                        content_blocks[current_tool_index]["name"] = tool_call.function.name
                        
                        # Send name update if changed
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_tool_index, 'delta': {'type': 'tool_use_delta', 'name': tool_call.function.name}})}\n\n"
                    
                    # Update tool arguments if provided
                    if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                        try:
                            # Parse the JSON arguments
                            args_json = json.loads(tool_call.function.arguments)
                            
                            # Update the input field in the block
                            content_blocks[current_tool_index]["input"] = args_json
                            
                            # Send input delta
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_tool_index, 'delta': {'type': 'tool_use_delta', 'input': args_json}})}\n\n"
                        except Exception as e:
                            # If we can't parse it as JSON, just use the string
                            content_blocks[current_tool_index]["input"] = tool_call.function.arguments
                            
                            # Send input delta
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_tool_index, 'delta': {'type': 'tool_use_delta', 'input': tool_call.function.arguments}})}\n\n"
            
            # Check if this is the final chunk
            if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason and not has_sent_stop_reason:
                # Close any open content blocks
                if tool_index is not None:
                    # Close all tool blocks first
                    for i in range(1, last_tool_index + 1):
                        if i in content_blocks and not text_block_closed:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                    
                    # Close the text content block if it wasn't closed before
                    if 0 in content_blocks and not text_block_closed:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                
                # Set flag to avoid duplicate stop messages
                has_sent_stop_reason = True
                
                # Map OpenAI finish reason to Anthropic stop reason
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason == "length":
                    stop_reason = "max_tokens"
                elif finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "stop":
                    stop_reason = "end_turn"
                else:
                    stop_reason = "end_turn"  # Default
                
                # Send message_delta with stop reason and usage
                usage = {"output_tokens": output_tokens}
                
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                
                # Send message_stop event
                message_stop_json = json.dumps({'type': 'message_stop'})
                yield f"event: message_stop\ndata: {message_stop_json}\n\n"
                
                # Send final [DONE] marker to match Anthropic's behavior
                yield "data: [DONE]\n\n"
                return
    
    except asyncio.CancelledError:
        # Handle client disconnection
        logger.info("Stream was cancelled (client disconnected)")
        # No need to yield anything more after the client disconnects
        return

    except GeneratorExit:
        # Handle generator exit explicitly (client disconnected)
        logger.info("Generator exited (client disconnected)")
        return
        
    except Exception as e:
        # Log error
        logger.error(f"Error processing chunk: {str(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        try:
            # Send error message_delta
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            
            # Send message_stop event
            message_stop_json = json.dumps({'type': 'message_stop'})
            yield f"event: message_stop\ndata: {message_stop_json}\n\n"
            
            # Send final [DONE] marker
            yield "data: [DONE]\n\n"
        except:
            # If we can't send the error response, just log it and return
            logger.error("Failed to send error response after exception")
            return
    
    finally:
        # Only try to yield final messages if we haven't sent a stop reason yet
        # and the generator hasn't been closed
        try:
            if not has_sent_stop_reason:
                # Try a zero-length yield to check if client is still connected
                try:
                    # This will raise GeneratorExit if client disconnected
                    yield ""
                    
                    # Continue with cleanup since client is still connected
                    # Close any open tool call blocks
                    if tool_index is not None:
                        for i in range(1, last_tool_index + 1):
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                    
                    # Close the text content block
                    if 0 in content_blocks and not text_block_closed:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    
                    # Send final message_delta with usage
                    usage = {"output_tokens": output_tokens}
                    
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
                    
                    # Send message_stop event
                    message_stop_json = json.dumps({'type': 'message_stop'})
                    yield f"event: message_stop\ndata: {message_stop_json}\n\n"
                    
                    # Send final [DONE] marker to match Anthropic's behavior
                    yield "data: [DONE]\n\n"
                    logger.info("Stream complete with normal termination in finally block")
                except GeneratorExit:
                    # Client already disconnected during our test yield
                    logger.info("Client disconnected during finally block cleanup")
                    # Re-raise to properly close the generator
                    raise
        except GeneratorExit:
            # Generator was exited during the finally block, which is fine
            logger.info("Generator exited during cleanup in finally block")
            # Re-raise to properly close the generator
            raise
        except Exception as e:
            # Log any errors in the finally block but don't re-raise
            logger.error(f"Error during stream cleanup: {str(e)}")
            pass

def mask_api_key(api_key):
    """Mask an API key for displaying in logs."""
    if not api_key:
        return "Not set"
    if len(api_key) <= 8:
        return "***" + api_key[-2:]
    return api_key[:4] + "..." + api_key[-4:]


def convert_message_to_openai(message):
    """Convert an Anthropic message to OpenAI format."""
    if not message:
        return {"role": "user", "content": ""}
    
    role = message.role
    content = ""
    
    # Handle content blocks
    if message.content and isinstance(message.content, list):
        for block in message.content:
            if block.type == "text":
                content += block.text
            # Handle other block types if needed
    
    # Return the converted message
    return {
        "role": role,
        "content": content
    }


async def handle_openai_streaming(openai_stream, request):
    """
    Handle streaming responses from OpenAI API.
    Handles the streaming format from OpenAI v1 client for Together.ai
    """
    message_id = f"msg_{uuid.uuid4()}"
    first_token_received = False
    content_block_index = 0
    current_content = "" # Holds content for the current text block (index 0)
    current_tool_calls = {} # Stores aggregated tool call info by *tool_index*
    stream_completed_successfully = False
    active_block_type = None # Track if the currently open block is 'text' or 'tool'
    
    # Special tracking for Together.ai tool detection from text
    is_together_model = request.model.startswith("together_ai/")
    has_tools = request.tools and len(request.tools) > 0
    accumulated_text = "" if (is_together_model and has_tools) else None
    detected_tool_calls = []
    text_block_closed = False

    try:
        # The OpenAI v1 client returns an iterator
        for chunk in openai_stream:
            if log_level <= logging.DEBUG or RICH_DEBUG:
                try:
                    chunk_json = chunk.model_dump_json(indent=2)
                    debug_console.print(f"[dim]Received chunk: {chunk_json}[/dim]")
                except Exception as json_err:
                    debug_console.print(f"[dim]Received chunk (non-JSON serializable?): {chunk} / Error: {json_err}[/dim]")

            # Process choices in the chunk
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                
                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta
                    
                    # --- Start Message --- 
                    if not first_token_received:
                        first_token_received = True
                        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': request.model}})}\n\n"
                    
                    # --- Handle Text Delta --- 
                    if hasattr(delta, "content") and delta.content is not None:
                        # If this is the first text piece and no block is active, start text block
                        if active_block_type is None:
                            active_block_type = 'text'
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text'}})}\n\n"
                        
                        # Ensure we are in a text block before adding content
                        if active_block_type == 'text':
                            current_content += delta.content
                            
                            # Accumulate text for Together.ai tool detection
                            if is_together_model and has_tools:
                                accumulated_text += delta.content
                                
                                # Check for complete tool call patterns in accumulated text
                                complete_json_block = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", accumulated_text, re.DOTALL)
                                complete_function_call = re.search(r"USE_TOOL:\s*(\w+)\s*PARAMETERS:\s*(\{.*?\})", accumulated_text, re.DOTALL)
                                
                                if (complete_json_block or complete_function_call) and not detected_tool_calls:
                                    # Attempt to detect tool calls
                                    tool_calls_found = detect_tool_calls_in_text(accumulated_text)
                                    
                                    if tool_calls_found:
                                        # Found tool calls in the text output
                                        detected_tool_calls = tool_calls_found
                                        
                                        # Close the current text block
                                        text_block_closed = True
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                                        
                                        # Create cleaned text without tool calls
                                        cleaned_text = accumulated_text
                                        cleaned_text = re.sub(r"```(?:json)?\s*\n.*?\n```", "[Tool Call Removed]", cleaned_text, flags=re.DOTALL)
                                        cleaned_text = re.sub(r"USE_TOOL:\s*\w+\s*PARAMETERS:\s*\{.*?\}", "[Tool Call Removed]", cleaned_text, flags=re.DOTALL)
                                        cleaned_text = re.sub(r"use the \w+ tool with parameters:?\s*\{.*?\}", "[Tool Call Removed]", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
                                        
                                        # Update current_content with cleaned text
                                        current_content = cleaned_text.strip()
                                        
                                        # Process each detected tool call
                                        for i, tool_call in enumerate(detected_tool_calls):
                                            tool_index = i + 1  # Start at 1 since 0 is text
                                            
                                            # Extract tool info
                                            function = tool_call.get('function', {})
                                            name = function.get('name', '')
                                            tool_id = tool_call.get('id', f"tool_{uuid.uuid4().hex[:8]}")
                                            arguments_str = function.get('arguments', '{}')
                                            
                                            try:
                                                arguments = json.loads(arguments_str)
                                            except json.JSONDecodeError:
                                                arguments = {"raw": arguments_str}
                                            
                                            # Start a new tool_use block
                                            content_block_data = {
                                                'type': 'content_block_start',
                                                'index': tool_index,
                                                'content_block': {
                                                    'type': 'tool_use',
                                                    'id': tool_id,
                                                    'name': name,
                                                    'input': {}
                                                }
                                            }
                                            yield f"event: content_block_start\ndata: {json.dumps(content_block_data)}\n\n"
                                            
                                            # Send the tool input delta
                                            tool_delta = {
                                                'type': 'content_block_delta',
                                                'index': tool_index, 
                                                'delta': {
                                                    'type': 'tool_use_delta',
                                                    'input': arguments
                                                }
                                            }
                                            yield f"event: content_block_delta\ndata: {json.dumps(tool_delta)}\n\n"
                                            
                                            # Close the tool_use block
                                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"
                                        
                                        # Set finish indicator
                                        stream_completed_successfully = True
                                        
                                        # Send final events
                                        message_delta_json = json.dumps({
                                            'type': 'message_delta',
                                            'delta': {'stop_reason': 'tool_use'},
                                            'usage': {'output_tokens': len(accumulated_text) // 4}  # Rough estimate
                                        })
                                        yield f"event: message_delta\ndata: {message_delta_json}\n\n"
                                        
                                        # Send message_stop event
                                        message_stop_json = json.dumps({'type': 'message_stop'})
                                        yield f"event: message_stop\ndata: {message_stop_json}\n\n"
                                        
                                        # Send final [DONE] marker
                                        yield "data: [DONE]\n\n"
                                        return
                            
                            # Send regular content delta if no tool calls detected and we're in a text block
                            if not text_block_closed and active_block_type == 'text':
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': delta.content}, 'index': content_block_index})}\n\n"
                            # Don't log warning about text in tool block - this is normal in some models
                            # and just creates log spam while not being actionable
                    
                    # --- Handle Tool Call Delta from native OpenAI format --- 
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            tool_index = tool_call_chunk.index # This is the index within the tool_calls list
                            if tool_index is None: continue # Skip if no index
                            
                            # Calculate the Anthropic content_block index for this tool
                            # It's 1 + tool_index if a text block exists, otherwise tool_index
                            anthropic_tool_block_index = tool_index + (1 if current_content or active_block_type == 'text' else 0)

                            # Start of a new tool call instance
                            if tool_call_chunk.id and tool_call_chunk.type == 'function':
                                # Close the text block if it was active
                                if active_block_type == 'text':
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                                    active_block_type = 'tool' # Switch active block type
                                    content_block_index = anthropic_tool_block_index # Update the main index counter
                                elif active_block_type is None:
                                    active_block_type = 'tool'
                                    content_block_index = anthropic_tool_block_index
                                
                                # Initialize storage for this tool call
                                if tool_index not in current_tool_calls:
                                    current_tool_calls[tool_index] = {
                                        "id": tool_call_chunk.id,
                                        "name": getattr(tool_call_chunk.function, 'name', ""), # Get initial name if available
                                        "input": getattr(tool_call_chunk.function, 'arguments', "") # Get initial args if available
                                    }
                                    # Send content_block_start for the tool
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_block_index, 'content_block': {'type': 'tool_use', 'id': tool_call_chunk.id, 'name': current_tool_calls[tool_index]['name'], 'input': {}}})}\n\n"
                                else:
                                     # This case might indicate receiving the ID again for an existing index? Log a warning.
                                     logger.warning(f"Received tool start chunk again for tool_index {tool_index}")

                            # Delta for an existing tool call (name or arguments)
                            if tool_call_chunk.function and tool_index in current_tool_calls:
                                delta_payload = {"type": "tool_use_delta"}
                                updated = False
                                if hasattr(tool_call_chunk.function, 'name') and tool_call_chunk.function.name:
                                    current_tool_calls[tool_index]["name"] = tool_call_chunk.function.name
                                    delta_payload["name"] = tool_call_chunk.function.name
                                    updated = True
                                if hasattr(tool_call_chunk.function, 'arguments') and tool_call_chunk.function.arguments:
                                    current_tool_calls[tool_index]["input"] += tool_call_chunk.function.arguments
                                    delta_payload["input"] = tool_call_chunk.function.arguments # Send only the delta part
                                    updated = True
                                
                                if updated:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_block_index, 'delta': delta_payload})}\n\n"
                                        
                # --- End of Message / Stop Reason --- 
                if hasattr(choice, "finish_reason") and choice.finish_reason is not None and not stream_completed_successfully:
                    stream_completed_successfully = True
                    stop_reason = "end_turn"
                    if choice.finish_reason == "length": stop_reason = "max_tokens"
                    elif choice.finish_reason == "tool_calls": stop_reason = "tool_use"
                        
                    # Close the last active content block (text or tool)
                    if active_block_type == 'text':
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                    elif active_block_type == 'tool':
                        # Calculate the index of the last tool block that was started
                        last_tool_anthropic_index = len(current_tool_calls) -1 + (1 if any(t['id'] for t in current_tool_calls.values()) and current_content == "" else 0) # Complicated... relies on text block presence
                        last_tool_anthropic_index = max(0, last_tool_anthropic_index) # Ensure non-negative index
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': last_tool_anthropic_index})}\n\n"
                        
                    # Log any parsing errors for the final tool inputs (don't send in stream)
                    for idx, tool_info in current_tool_calls.items():
                        if isinstance(tool_info["input"], str):
                            try: json.loads(tool_info["input"])
                            except json.JSONDecodeError as json_err:
                                logger.warning(f"Final tool input JSON invalid for tool {tool_info.get('name', '?')} (id: {tool_info.get('id', '?')}): {json_err}")

                    # Send final message events
                    message_delta_json = json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}})
                    yield f"event: message_delta\ndata: {message_delta_json}\n\n"
                    
                    message_stop_json = json.dumps({'type': 'message_stop'})
                    yield f"event: message_stop\ndata: {message_stop_json}\n\n"
                    
                    yield f"data: [DONE]\n\n"
                    return # Clean exit from generator
    
    except asyncio.CancelledError:
        # Handle client disconnection properly
        logger.info("Stream was cancelled (client disconnected)")
        return
        
    except GeneratorExit:
        # Handle generator exit explicitly - immediately terminate without any further yields
        logger.info("Generator exited (client disconnected)")
        # Must exit cleanly and immediately without any further operations
        # DO NOT call any other yield operations after receiving GeneratorExit
        return

    except Exception as e:
        import traceback
        error_message = f"Error during Together.ai stream processing: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        try:
            error_payload = json.dumps({
                'type': 'error',
                'error': {'type': 'internal_server_error', 'message': error_message}
            })
            yield f"event: error\ndata: {error_payload}\n\n"
        except Exception as yield_err: 
            logger.error(f"Failed to yield error event: {yield_err}")
            return

    finally:
        # For cleanup operations that don't involve yielding
        # The key is to NOT yield anything if stream_completed_successfully is False,
        # as we might be in the middle of handling a GeneratorExit
        
        # We can log information, but must not yield after GeneratorExit
        if not stream_completed_successfully:
            logger.info("Stream terminated without completion")
        
        # Any cleanup that doesn't involve yield operations can go here
        # DO NOT attempt to yield in a finally block after GeneratorExit
        # This avoids the "async generator ignored GeneratorExit" exception

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
    background_tasks: BackgroundTasks,
    response: Response,
):
    """
    Create a message (i.e., perform inference).
    """
    try:
        # print the body here for debugging
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        elif clean_model.startswith("gemini/"):
            clean_model = clean_model[len("gemini/"):]
        elif clean_model.startswith("together_ai/"):
            clean_model = clean_model[len("together_ai/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Special case: If using Together.ai model, use the OpenAI client directly
        if request.model.startswith("together_ai/"):
            # Extract model name (already done in convert_anthropic_to_litellm)
            model_name = litellm_request["model"]
            
            # Initialize OpenAI client
            client = openai.OpenAI(
                api_key=TOGETHERAI_API_KEY,
                base_url="https://api.together.xyz/v1",
            )
            
            # Log request information
            logger.info(f"Using direct OpenAI client for Together.ai model: {model_name}")
            
            # Create request parameters - convert from LiteLLM format to OpenAI params
            openai_params = {
                "model": model_name,
                "messages": litellm_request["messages"],
                "stream": request.stream,
            }
            
            # Add optional parameters
            if "max_tokens" in litellm_request:
                openai_params["max_tokens"] = litellm_request["max_tokens"]
            if "temperature" in litellm_request:
                openai_params["temperature"] = litellm_request["temperature"]
            if "top_p" in litellm_request:
                openai_params["top_p"] = litellm_request["top_p"]
            if "stop" in litellm_request:
                openai_params["stop"] = litellm_request["stop"]
                
            # Add tools if they exist
            if "tools" in litellm_request:
                openai_params["tools"] = litellm_request["tools"]
                
            # Add tool_choice if it exists
            if "tool_choice" in litellm_request:
                openai_params["tool_choice"] = litellm_request["tool_choice"]
            
            # Debug log the request
            if log_level <= logging.DEBUG or RICH_DEBUG:
                debug_message = f"Together.ai request with {len(openai_params['messages'])} messages"
                if 'tools' in openai_params:
                    debug_message += f" and {len(openai_params['tools'])} tools"
                debug_console.print(Panel(
                    debug_message,
                    title="[bold]Together.ai Direct API Call[/bold]",
                    border_style="green"
                ))
            
            try:
                # Make API call based on streaming preference
                if request.stream:
                    # Streaming call
                    stream = client.chat.completions.create(**openai_params)
                    
                    # Return streaming response
                    return StreamingResponse(
                        handle_openai_streaming(stream, request),
                        media_type="text/event-stream"
                    )
                else:
                    # Non-streaming call
                    start_time = time.time()
                    completion = client.chat.completions.create(**openai_params)
                    response_time = time.time() - start_time
                    
                    logger.info(f"Together.ai response received in {response_time:.2f}s")
                    
                    # Convert to Anthropic format
                    content = []
                    
                    # Extract text content
                    message_content = completion.choices[0].message.content
                    if message_content:
                        content.append({"type": "text", "text": message_content})
                    
                    # Extract tool calls if present
                    tool_calls = completion.choices[0].message.tool_calls
                    if tool_calls:
                        logger.info(f"Tool calls received from Together.ai: {len(tool_calls)}")
                        
                        for tool_call in tool_calls:
                            # Extract function data
                            function = tool_call.function
                            tool_use = {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": function.name,
                                "input": json.loads(function.arguments)
                            }
                            content.append(tool_use)
                    
                    # Map finish reason to stop reason
                    stop_reason = "end_turn"
                    finish_reason = completion.choices[0].finish_reason
                    if finish_reason == "length":
                        stop_reason = "max_tokens"
                    elif finish_reason == "tool_calls":
                        stop_reason = "tool_use"
                    
                    # Create Anthropic-style response
                    anthropic_response = MessagesResponse(
                        id=completion.id,
                        model=request.model,  # Use original model name with prefix
                        role="assistant",
                        content=content,
                        stop_reason=stop_reason,
                        stop_sequence=None,
                        usage=Usage(
                            input_tokens=completion.usage.prompt_tokens,
                            output_tokens=completion.usage.completion_tokens
                        )
                    )
                    
                    return anthropic_response
                    
            except Exception as e:
                # Log the error
                logger.error(f"Error calling Together.ai API: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Together.ai API error: {str(e)}")
        
        # Normal flow for other models using LiteLLM
        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        
        try:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Handle streaming mode
            if request.stream:
                # Use LiteLLM for streaming
                response_generator = await litellm.acompletion(**litellm_request)
                
                return StreamingResponse(
                    handle_streaming(response_generator, request),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                litellm_response = litellm.completion(**litellm_request)
                
                # Convert LiteLLM response to Anthropic format
                anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
                
                return anthropic_response
                
        except Exception as e:
            logger.error(f"Error with LiteLLM API call: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error with model API: {str(e)}")
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Log all error details
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        
        # Return detailed error
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        elif clean_model.startswith("gemini/"):
            clean_model = clean_model[len("gemini/"):]
        elif clean_model.startswith("together_ai/"):
            clean_model = clean_model[len("together_ai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Special handling for Together.ai models
            if request.model.startswith("together_ai/"):
                # Get the provider name for debug messages
                provider_name = "Together.ai"
                
                # Similar to create_message function, update the request for Together.ai
                converted_request["custom_llm_provider"] = "together_ai"
                converted_request["base_url"] = "https://api.together.xyz/v1"
                # Extract model name without provider prefix
                model_name = converted_request["model"]
                if model_name.startswith("together_ai/"):
                    model_name = model_name[len("together_ai/"):]
                    converted_request["model"] = model_name
                
                # Rich debug information for token counting
                if log_level <= logging.DEBUG or RICH_DEBUG:
                    debug_console.print(Panel(
                        f"[bold blue]{provider_name} Token Counting[/bold blue]\n"
                        f"Original model: [cyan]{request.model}[/cyan]\n"
                        f"Converted model: [cyan]{model_name}[/cyan]\n"
                        f"Base URL: {converted_request['base_url']}\n"
                        f"Custom Provider: {converted_request['custom_llm_provider']}",
                        title="[bold]Token Count Configuration[/bold]",
                        border_style="cyan"
                    ))
                    
                    # Check for potential issues
                    warnings = []
                    if not TOGETHERAI_API_KEY:
                        warnings.append(f"[bold red]{provider_name} API key is not set![/bold red]")
                    
                    if warnings:
                        debug_console.print("[bold yellow]⚠️ Warnings:[/bold yellow]")
                        for warning in warnings:
                            debug_console.print(f" • {warning}")
                
                # Use OpenAI client directly when preferred provider is "together"
                if PREFERRED_PROVIDER == "together":
                    try:
                        # Initialize the OpenAI client with Together.ai endpoint
                        openai_client = openai.OpenAI(
                            base_url=converted_request["base_url"],
                            api_key=TOGETHERAI_API_KEY
                        )
                        
                        # Convert messages format for OpenAI
                        openai_request = {
                            "model": converted_request["model"],
                            "messages": converted_request["messages"]
                        }
                        
                        if log_level <= logging.DEBUG or RICH_DEBUG:
                            debug_console.print(Panel(
                                f"[bold green]Using OpenAI client directly for {provider_name} token counting[/bold green]",
                                border_style="green"
                            ))
                        
                        # Use client token counting method
                        token_count_response = openai_client.chat.completions.create(**openai_request, max_tokens=0)
                        token_count = token_count_response.usage.prompt_tokens
                        
                        # Return the token count
                        return TokenCountResponse(input_tokens=token_count)
                    except Exception as e:
                        logger.error(f"Error using OpenAI client for token counting: {str(e)}")
                        # Fall back to LiteLLM token counting
                        logger.info("Falling back to LiteLLM token counting")
            
            # Count tokens
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        print("Additional options:")
        print("  --debug-rich    Enable rich console debugging output")
        print("  --process-newlines  Process newlines in Together.ai responses")
        print("  --preserve-openai-newlines  Keep newlines as-is in OpenAI client responses")
        sys.exit(0)
    
    # Check for rich debug flag
    if "--debug-rich" in sys.argv:
        # Configure rich console to show debugging info regardless of log level
        debug_console.print(Panel(
            "[bold green]Rich debugging enabled[/bold green]\n"
            "You will see detailed debugging information for API calls, especially for Together.ai",
            title="[bold]Debug Mode[/bold]",
            border_style="green"
        ))
        # Override log level for rich console
        RICH_DEBUG = True
    else:
        RICH_DEBUG = False
    
    # Check for newline processing flag
    if "--process-newlines" in sys.argv:
        PRESERVE_TOGETHER_FORMATTING = False
        print("Together.ai newline processing enabled - newlines will be transformed")
    
    # Check for OpenAI newline preservation flag
    if "--preserve-openai-newlines" in sys.argv:
        PRESERVE_OPENAI_NEWLINES = True
        print("OpenAI newline preservation enabled - newlines will be kept as-is")
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")