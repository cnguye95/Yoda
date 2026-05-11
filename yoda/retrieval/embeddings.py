"""Converts text into embedding vectors using the OpenAI API.

We use text-embedding-3-small for all embeddings. It produces 1536-dimensional
vectors, scores well on the MTEB retrieval benchmark, and costs roughly
$0.004 per full 10-Q filing — cheap enough to re-embed whenever needed.
The openai package (already pinned in requirements.txt) is the only dependency.

The one public function, embed_texts(), accepts a list of strings and returns
a parallel list of 1536-element float lists in the same order as the input.
It batches the API calls at 100 texts per call to keep individual requests
small and the response size manageable.
"""

from openai import OpenAI

from yoda import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The embedding model and the vector dimension it produces.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Maximum number of texts to send in a single API call. OpenAI accepts up to
# 2048; we use 100 to keep responses small and progress easy to observe.
_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused for every call)
# ---------------------------------------------------------------------------

# Instantiate the client here so it is shared across all embed_texts() calls
# within a Python process — avoids re-reading the API key on every call.
_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a 1536-dim embedding vector for each string in *texts*.

    The output list has the same length and order as the input. Each element
    is a list of 1536 floats suitable for cosine-similarity search.

    Parameters
    ----------
    texts : list[str]
        Strings to embed. Empty strings are allowed but will produce a
        near-zero vector that won't retrieve well — callers should filter
        them out before embedding.
    """
    if not texts:
        return []

    results: list[list[float]] = []

    # Process texts in batches of _BATCH_SIZE to keep API calls small.
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]

        # Call the OpenAI embeddings endpoint. The response data list is
        # guaranteed to be in the same order as the input batch.
        response = _client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        # Extract the float vectors from the response objects and extend our
        # running results list so the final order matches the input order.
        for item in response.data:
            results.append(item.embedding)

    return results
