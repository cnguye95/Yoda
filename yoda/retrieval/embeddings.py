"""Converts text into embedding vectors using a selectable provider.

Two providers are supported:

  - "openai" (default): text-embedding-3-small via the OpenAI API.
    1536-dim vectors, ~$0.004 per 10-Q filing, very fast.

  - "qwen": Qwen3-Embedding-0.6B via sentence-transformers, run locally.
    1024-dim vectors, no API cost or rate limits, ~50-200ms per batch on CPU.
    Requires a one-time ~1.2 GB model download on first use.

embed_texts(texts, provider="openai") dispatches to the right backend.
Vectors from one provider are NOT comparable to vectors from the other —
the ChromaStore writes each provider's vectors into a separate collection.
"""

from openai import OpenAI

from yoda import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OpenAI model + dimension (existing).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Qwen model + dimension (new). Qwen3-Embedding-0.6B picked for size/quality
# balance — 1024-dim is enough for retrieval, model is ~1.2 GB on disk.
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DIMENSION = 1024

# Maximum number of texts to send in a single OpenAI API call.
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused for every call)
# ---------------------------------------------------------------------------

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# Lazy-loaded Qwen model — only loaded when first needed so OpenAI-only users
# never pay the import cost of sentence-transformers/torch.
_qwen_model = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], provider: str = "openai") -> list[list[float]]:
    """Return an embedding vector for each string in *texts*.

    The output list has the same length and order as the input.

    Parameters
    ----------
    texts : list[str]
        Strings to embed. Empty list returns an empty list.
    provider : str
        "openai" (default) for text-embedding-3-small via API, or
        "qwen" for Qwen3-Embedding-0.6B run locally.
    """
    if not texts:
        return []
    if provider == "qwen":
        return _embed_qwen(texts)
    return _embed_openai(texts)


# ---------------------------------------------------------------------------
# OpenAI backend — text-embedding-3-small via API, batched at 100/call
# ---------------------------------------------------------------------------

def _embed_openai(texts: list[str]) -> list[list[float]]:
    # Batched calls keep individual responses small and progress observable.
    results: list[list[float]] = []
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        response = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            results.append(item.embedding)
    return results


# ---------------------------------------------------------------------------
# Qwen backend — Qwen3-Embedding-0.6B via sentence-transformers, run locally
# ---------------------------------------------------------------------------

def _embed_qwen(texts: list[str]) -> list[list[float]]:
    # Lazy-load the model on first use. sentence-transformers caches it in
    # ~/.cache/huggingface/ so subsequent runs in a fresh process skip the download.
    global _qwen_model
    if _qwen_model is None:
        from sentence_transformers import SentenceTransformer
        _qwen_model = SentenceTransformer(QWEN_MODEL)

    # encode() returns a numpy array; .tolist() gives plain Python lists matching the OpenAI signature.
    embeddings = _qwen_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()
