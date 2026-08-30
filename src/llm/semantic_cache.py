"""Opt-in local embeddings for safe semantic LLM-cache lookup.

The cache client owns persistence; this module only defines the request
contract and the encoder. Callers must put every exact context dependency
(evidence, tables, system prompt, etc.) in ``guard``. Only ``text`` is allowed
to vary semantically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class SemanticCacheRequest:
    """The one text allowed to vary plus context that must match exactly."""

    text: str
    guard: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("semantic cache text cannot be empty")


class SemanticEncoder(Protocol):
    """Small interface so tests do not load or download an embedding model."""

    @property
    def model_id(self) -> str: ...

    def encode(self, text: str) -> Sequence[float]: ...


class SentenceTransformerSemanticEncoder:
    """Lazy multilingual sentence-transformer encoder.

    Loading is deferred until the first eligible request. Semantic cache is
    disabled by default, so ordinary CLI/test use never downloads a model or
    pays its cold-start cost.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self.model_name

    def encode(self, text: str) -> Sequence[float]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "semantic cache needs sentence-transformers; "
                    "run pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=False,
            )

        vector = self._model.encode(
            [f"query: {text.strip()}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return [float(value) for value in vector]
