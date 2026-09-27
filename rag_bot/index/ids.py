"""Deterministic collection naming (architecture D5).

The collection name encodes the embedding model AND the corpus version, so
changing either produces a new collection rather than silently mixing
incompatible vectors. Chroma collection names allow [a-zA-Z0-9._-] only, so both
parts are slugified.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

COLLECTION_PREFIX = "mf_faq"
_ALLOWED = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _ALLOWED.sub("-", text).strip("-.")
    return text.lower() or "x"


def collection_name(embed_model: str, corpus_version: str) -> str:
    """'mf_faq__<slug(embed_model)>__<corpus_version>' (architecture D5).

    A short digest is appended when the slugified model name is long, so the
    name stays under Chroma's 512-char limit without collisions.
    """
    model_slug = slugify(embed_model)
    if len(model_slug) > 48:
        digest = hashlib.blake2b(
            embed_model.encode("utf-8"), digest_size=4
        ).hexdigest()
        model_slug = f"{model_slug[:40]}-{digest}"
    version_slug = slugify(corpus_version)
    return f"{COLLECTION_PREFIX}__{model_slug}__{version_slug}"
