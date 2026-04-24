import logging
import os

def silence():
    # 1. Global logging level (only show warnings and errors)
    logging.basicConfig(level=logging.WARNING)

    # 2. Silence noisy libraries individually
    for logger_name in [
        "httpx",
        "httpcore",
        "chromadb",
        "sentence_transformers",
        "urllib3",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # 3. Optional: completely disable debug logs from asyncio / others
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # 4. Silence environment-based verbose modes (important for transformers)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"