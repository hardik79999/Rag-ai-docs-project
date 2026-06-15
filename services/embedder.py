# FIXED:
# 1. Model changed to "gemini-embedding-001" — only embedding model confirmed
#    available on this API key via client.models.list()
# 2. text-embedding-004 removed — NOT available on this key (404 error)
# 3. All calls use new google-genai SDK: client.models.embed_content()
# 4. Result extracted via result.embeddings[0].values
# 5. ASYNC UPDATE: Moved to client.aio for non-blocking FastAPI performance.

import asyncio
import logging
from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)
_client = genai.Client(api_key=settings.gemini_api_key)

# Confirmed available on this API key via ModelService.ListModels
EMBEDDING_MODEL = "gemini-embedding-001"


async def get_embedding(text: str) -> list[float]:
    """Single document chunk ka embedding lo (RETRIEVAL_DOCUMENT task)."""
    result = await _client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


async def get_query_embedding(text: str) -> list[float]:
    """User query ka embedding lo (RETRIEVAL_QUERY task).

    Different task_type than documents — improves retrieval accuracy.
    """
    result = await _client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


async def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Multiple document chunks ke embeddings generate karo."""
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_retries = 5
        base_delay = 15
        
        for attempt in range(max_retries):
            try:
                result = await _client.aio.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                for emb in result.embeddings:
                    embeddings.append(emb.values)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        raise e
                else:
                    raise e
    return embeddings
