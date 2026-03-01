import ollama

# Model config
LLM_MODEL = "gemma2:9b"
EMBED_MODEL = "mxbai-embed-large"

# Memory config
STM_MAX_TURNS = 15
EPISODIC_RETRIEVE_K = 10
FACTS_RETRIEVE_K = 10
SUMMARY_UPDATE_EVERY_N_TURNS = 7
BOOST_AMOUNT = 0.1

# Paths
CHROMA_PATH = "./chroma_db"
COMPACT_MEMORY_PATH = "./compact_memory.json"


def embed(text: str) -> list:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]


def llm_chat(messages: list) -> str:
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]