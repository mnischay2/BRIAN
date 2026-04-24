import asyncio
import os
import ollama
try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from scripts.configs.config import CONF

# Persistent async clients
_ollama_client = None
_gemini_client = None

def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.AsyncClient(host=CONF.get("LLM_BASE_URL", "http://localhost:11434"))
    return _ollama_client

def _get_gemini_client():
    """Initialize Gemini client with API key."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_AVAILABLE:
            raise ImportError("google.genai not installed. Install with: pip install google-genai")
        api_key = CONF.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or config")
        # Set API key as environment variable for google.genai
        os.environ["GOOGLE_API_KEY"] = api_key
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def chat_snt_off(message):
    """Synchronous chat (for non-async contexts). Supports both local Ollama and cloud Gemini."""
    llm_usage = CONF.get("LLM_USAGE", "local").lower()
    
    if llm_usage == "cloud":
        # Use Gemini Cloud
        client = _get_gemini_client()
        # Convert message format for Gemini
        messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in message])
        model_name = CONF.get("GEMINI_MODEL", "gemini-2.0-flash")
        response = client.models.generate_content(
            model=model_name,
            contents=messages_text
        )
        return response.text
    else:
        # Use Ollama (default)
        response = ollama.chat(
            model=CONF["LLM_MODEL"],
            messages=message,
            stream=False,
        )
        return response["message"]["content"]


async def async_chat_snt_off(message, max_retries=3, initial_delay=0.5):
    """
    Call LLM (local Ollama or cloud Gemini) with exponential backoff retry logic.
    
    Parameters:
    -----------
    message : list[dict]
        Messages in [{"role": "...", "content": "..."}] format
    max_retries : int
        Maximum number of retry attempts (default 3)
    initial_delay : float
        Initial retry delay in seconds (default 0.5)
    
    Returns:
    --------
    str : LLM response content
    """
    llm_usage = CONF.get("LLM_USAGE", "local").lower()
    delay = initial_delay
    
    if llm_usage == "cloud":
        # Use Gemini Cloud (synchronous API wrapped in async)
        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                client = _get_gemini_client()
                model_name = CONF.get("GEMINI_MODEL", "gemini-2.0-flash")
                # Convert message format for Gemini
                messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in message])
                # Run Gemini call in executor to avoid blocking
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=messages_text
                    )
                )
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
    else:
        # Use Ollama (default)
        client = _get_ollama_client()
        for attempt in range(max_retries):
            try:
                response = await client.chat(
                    model=CONF["LLM_MODEL"],
                    messages=message,
                    stream=False,
                )
                return response["message"]["content"]
            except (ollama._types.ResponseError, asyncio.TimeoutError, EOFError) as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as e:
                raise

client = _get_gemini_client()

for m in client.models.list():
    print(m.name)