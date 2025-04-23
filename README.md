# Anthropic API Proxy for Gemini & OpenAI Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, or Together.ai backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, or Together.ai models via LiteLLM. 🌉

# NOTE: Together AI detects tools, returns response but it gets converted to text? not working currently.

![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Google AI Studio (Gemini) API key (if using Google provider) 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys and model configurations:

   *   `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models.
   *   `OPENAI_API_KEY`: Your OpenAI API key (Required if using the default OpenAI preference or as fallback).
   *   `GEMINI_API_KEY`: Your Google AI Studio (Gemini) API key (Required if PREFERRED_PROVIDER=google).
   *   `PREFERRED_PROVIDER` (Optional): Set to `openai` (default), `google`, or `together`. This determines the primary backend for mapping `haiku`/`sonnet`.
   *   `BIG_MODEL` (Optional): The model to map `sonnet` requests to. Defaults to `gpt-4.1` (if `PREFERRED_PROVIDER=openai`) or `gemini-2.5-pro-preview-03-25`.
   *   `SMALL_MODEL` (Optional): The model to map `haiku` requests to. Defaults to `gpt-4.1-mini` (if `PREFERRED_PROVIDER=openai`) or `gemini-2.0-flash`.

   **Mapping Logic:**
   - If `PREFERRED_PROVIDER=openai` (default), `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `openai/`.
   - If `PREFERRED_PROVIDER=google`, `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `gemini/` *if* those models are in the server's known `GEMINI_MODELS` list (otherwise falls back to OpenAI mapping).

4. **Run the server**:
   ```bash
   ./run-server.sh
   ```
   *(the script sets sensible defaults, including `--reload`, and can be customised by editing environment variables inside)*

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models (defaulting to Gemini) through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude models to either OpenAI or Gemini models based on the configured model:

| Claude Model | Default Mapping | When BIG_MODEL/SMALL_MODEL is a Gemini model |
|--------------|--------------|---------------------------|
| haiku | openai/gpt-4o-mini | gemini/[model-name] |
| sonnet | openai/gpt-4o | gemini/[model-name] |

### Supported Models

#### OpenAI Models
The following OpenAI models are supported with automatic `openai/` prefix handling:
- o3-mini
- o1
- o1-mini
- o1-pro
- gpt-4.5-preview
- gpt-4o
- gpt-4o-audio-preview
- chatgpt-4o-latest
- gpt-4o-mini
- gpt-4o-mini-audio-preview
- gpt-4.1
- gpt-4.1-mini

#### Gemini Models
The following Gemini models are supported with automatic `gemini/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash

### Model Prefix Handling
The proxy automatically adds the appropriate prefix to model names:
- OpenAI models get the `openai/` prefix 
- Gemini models get the `gemini/` prefix
- The BIG_MODEL and SMALL_MODEL will get the appropriate prefix based on whether they're in the OpenAI or Gemini model lists

For example:
- `gpt-4o` becomes `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25`
- When BIG_MODEL is set to a Gemini model, Claude Sonnet will map to `gemini/[model-name]`

### Customizing Model Mapping

Control the mapping using environment variables in your `.env` file or directly:

**Example 1: Default (Use OpenAI)**
No changes needed in `.env` beyond API keys, or ensure:
```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key" # Needed if PREFERRED_PROVIDER=google
# PREFERRED_PROVIDER="openai" # Optional, it's the default
# BIG_MODEL="gpt-4.1" # Optional, it's the default
# SMALL_MODEL="gpt-4.1-mini" # Optional, it's the default
```

**Example 2: Prefer Google**
```dotenv
GEMINI_API_KEY="your-google-key"
OPENAI_API_KEY="your-openai-key" # Needed for fallback
PREFERRED_PROVIDER="google"
# BIG_MODEL="gemini-2.5-pro-preview-03-25" # Optional, it's the default for Google pref
# SMALL_MODEL="gemini-2.0-flash" # Optional, it's the default for Google pref
```

**Example 3: Use Specific OpenAI Models**
```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key"
PREFERRED_PROVIDER="openai"
BIG_MODEL="gpt-4o" # Example specific model
SMALL_MODEL="gpt-4o-mini" # Example specific model
```

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to OpenAI format via LiteLLM 🔄
3. **Sending** the translated request to OpenAI 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁

# Claude to OpenAI/Gemini/Together.ai Proxy

This proxy service translates Claude API requests to OpenAI, Gemini, or Together.ai API requests, allowing you to use Claude-compatible code with other LLM providers.

## Setup

1. Create a `.env` file with your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
TOGETHER_API_KEY=your_together_key

# Set your preferred provider (openai, google, or together)
PREFERRED_PROVIDER=together

# Choose which models to use
# BIG_MODEL=gpt-4.1
# SMALL_MODEL=gpt-4.1-mini
# TOGETHER_MODEL=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
```

2. Run the server:

```
uvicorn server:app --reload --host 0.0.0.0 --port 8082
```

## Using with Together.ai

Setting `PREFERRED_PROVIDER=together` in your `.env` file will route Claude requests to Together.ai using the Llama-4-Maverick model by default. You can specify a different Together.ai model by setting the `TOGETHER_MODEL` environment variable.

## Installation with Together.ai Support

Make sure to install the package with Together.ai support:

```bash
# If using pip
pip install -e "."

# If using uv
uv pip install -e "."

# Or simply use the run-server.sh script which handles this automatically
./run-server.sh
```

## Configuration

In your `.env` file:

```
TOGETHERAI_API_KEY=your_together_api_key
PREFERRED_PROVIDER=together
TOGETHER_MODEL=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
```

Note: Make sure to use `TOGETHERAI_API_KEY` (not TOGETHER_API_KEY) as this is what LiteLLM expects.

The server will map Claude models to Together.ai as follows:

- Claude Haiku → Together.ai model (from TOGETHER_MODEL)
- Claude Sonnet → Together.ai model (from TOGETHER_MODEL)

## Model Format

Together.ai models must be referenced using the format: `together_ai/model_name`

For example:
```
together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
```

The `together_ai/` prefix is essential - it tells LiteLLM which provider to use.

## Available Together.ai Models

Currently, the server is configured to support:

- meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 (default)

## Newline Handling

The server includes options to control how newlines are handled in model responses:

### Together.ai Newline Handling
By default, the server processes newlines in Together.ai responses for better readability. Literal `\n` characters in responses are converted to actual newlines.

### OpenAI Newline Handling
For OpenAI models, the server converts actual newlines to literal `\n` characters in the response to ensure proper display in OpenAI clients.

### Command Line Options

You can control newline handling behavior using these command-line flags when starting the server:

```bash
# Start the server with default newline handling
uvicorn server:app --reload --host 0.0.0.0 --port 8082

# Process newlines in Together.ai responses (enabled by default)
python server.py --process-newlines

# Keep actual newlines in OpenAI responses (don't convert to \n literals)
python server.py --preserve-openai-newlines

# View all available options
python server.py --help
```

Or use the run-server.sh script with options:

```bash
./run-server.sh --process-newlines --debug-rich
```

## Troubleshooting

If you encounter errors like:
- "Unmapped LLM provider for this endpoint"
- Authentication failures
- Rate limit issues

Make sure you:
1. Have a valid Together.ai API key set as `TOGETHERAI_API_KEY` (not TOGETHER_API_KEY)
2. Installed the package with Together.ai support (`pip install -e ".[together]"`)
3. Are using the correct model format with the `together_ai/` prefix
4. Are using a model that's available in your Together.ai plan
