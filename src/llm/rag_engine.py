"""
Retrieval-Augmented Generation (RAG) engine.

Loads the editable ``config/knowledge_base.json`` file and performs
**semantic search** to inject relevant product knowledge into the LLM
prompt at conversation time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOR BUSINESS USERS — How to add new Q&A without touching Python code:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  Open  config/knowledge_base.json  in any text editor.

2.  Find the section that makes sense for your new entry, for example
    "pricing" or "faq". (Add a new section if none fits — any top-level
    key whose value is an array of objects is automatically picked up.)

3.  Copy and paste an existing entry block, then change the values:

        {
          "key":      "my_new_topic",          ← unique ID (no spaces)
          "question": "What is X?",            ← the question the entry answers
          "answer":   "X is ...",              ← the full answer the bot gives
          "keywords": ["x", "topic", "thing"]  ← extra words to improve search
        }

4.  Save the file.

5.  Re-load without restarting the bot — choose any one of:
      a. Call  POST /reload/knowledge  on the API.
      b. Call  engine.reload()  in Python.
      c. Enable  auto_reload=True  (default) — the engine detects the
         file change automatically on the next customer question.

That's it! The semantic search model will pick up the new entry without
retraining or rebuilding anything.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THE SEMANTIC SEARCH WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Customer question  ──►  sentence-transformers model  ──►  384-dim vector
                                                                │
                                                       cosine similarity
                                                                │
                                         all KB entry vectors  ┘
                                                                │
                                              top-k best matches
                                                                │
                                        [KNOWLEDGE BASE CONTEXT] block
                                                                │
                                           injected into LLM system prompt

  Primary:  sentence-transformers  all-MiniLM-L6-v2
            Fast (~30 ms/query on CPU), accurate on short Q&A pairs.
            Install:  pip install sentence-transformers

  Fallback: TF-IDF cosine similarity (built-in, zero extra dependencies).
            Activated automatically if sentence-transformers is not installed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW CONTEXT IS INJECTED INTO THE LLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  system_prompt = engine.inject_context(base_system_prompt, customer_query)
  # → base prompt + appended KB context block passed to the LLM

  The LLM receives something like:

      You are a helpful insurance agent...

      [KNOWLEDGE BASE CONTEXT]
      --- Reference 1 (pricing) ---
      Q: How much does final expense insurance cost per month?
      A: Monthly premiums vary by age...
      [END CONTEXT]

      Use the above context to answer the customer's question accurately.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import RAGConfig, get_rag_config

logger = logging.getLogger(__name__)

# ── Try to import sentence-transformers; fall back to TF-IDF if absent ────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — falling back to TF-IDF search. "
        "Run  pip install sentence-transformers  for better retrieval quality."
    )

# Model name: small (80 MB), fast (CPU ~30 ms/query), strong on Q&A pairs.
# Change here to upgrade to a larger model — no other code needs updating.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved knowledge entry."""

    category: str   # section name from knowledge_base.json (e.g. "pricing")
    question: str   # the question this entry answers
    answer:   str   # the full answer text
    score:    float # similarity score 0–1 (higher = more relevant)


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """Semantic (or TF-IDF fallback) retrieval over ``knowledge_base.json``.

    Designed to require zero Python changes when the knowledge base is
    edited — just save the JSON and call ``reload()`` (or rely on
    ``auto_reload=True``).

    Quick start::

        engine = RAGEngine()
        engine.load()

        # Retrieve relevant chunks for a customer question
        chunks = engine.retrieve("How much does it cost per month?", top_k=3)

        # Format for display / logging
        print(engine.format_context(chunks))

        # Inject into a system prompt for the LLM
        prompt = engine.inject_context(base_system_prompt, customer_question)

    Parameters
    ----------
    config:
        Optional ``RAGConfig`` override.  Defaults to environment / .env.
    model_name:
        Sentence-transformers model name.  Defaults to all-MiniLM-L6-v2.
    auto_reload:
        When ``True`` (default), the engine checks the knowledge_base.json
        file modification time before each ``retrieve()`` call and
        reloads automatically if the file has changed.  Zero restarts needed.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        model_name: str = _DEFAULT_MODEL,
        auto_reload: bool = True,
    ) -> None:
        self._config     = config or get_rag_config()
        self._model_name = model_name
        self._auto_reload = auto_reload

        # Sentence-transformers state
        self._model           = None          # SentenceTransformer instance
        self._entry_embeddings = None         # np.ndarray shape (N, D)

        # TF-IDF fallback state
        self._idf:           dict[str, float] = {}
        self._entry_vectors: list[Counter]    = []

        # Shared state
        self._entries:    list[dict]  = []
        self._loaded:     bool        = False
        self._kb_mtime:   float       = 0.0   # last seen file mtime
        self._reload_lock = threading.Lock()  # prevent concurrent reloads

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def load(self, path: Optional[Path] = None) -> None:
        """Load and index the knowledge base from disk.

        Safe to call multiple times — fully rebuilds the index each time
        so edits to ``knowledge_base.json`` are always picked up.

        In production this takes ~1–2 seconds on first call (model load +
        embed all entries).  Subsequent ``reload()`` calls only re-embed
        entries (~200 ms for 50 entries on CPU).
        """
        kb_path = path or self._config.knowledge_base_path

        with self._reload_lock:
            self._load_locked(kb_path)

    def _load_locked(self, kb_path: Path) -> None:
        """Internal load — must be called while holding ``_reload_lock``."""
        # ── Parse JSON ────────────────────────────────────────────────────
        try:
            raw = json.loads(kb_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load knowledge base from %s", kb_path)
            raise

        self._kb_mtime = kb_path.stat().st_mtime

        # ── Flatten all array sections into a single entry list ───────────
        # Any top-level key whose value is a JSON array is treated as a
        # knowledge category. Non-array keys (like _README) are skipped.
        entries: list[dict] = []
        for category, items in raw.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                entries.append({
                    "category": category,
                    # Accept multiple JSON field name conventions so business
                    # users can use whatever feels natural
                    "question": item.get("question")
                                or item.get("objection")
                                or item.get("topic")
                                or "",
                    "answer":   item.get("answer")
                                or item.get("response")
                                or item.get("detail")
                                or "",
                    "keywords": [k.lower() for k in item.get("keywords", [])],
                })

        self._entries = entries

        # ── Build semantic index (sentence-transformers) ──────────────────
        if _SEMANTIC_AVAILABLE:
            self._build_semantic_index()
        else:
            self._build_tfidf_index()

        self._loaded = True
        logger.info(
            "RAG engine loaded %d entries from %s  (backend=%s)",
            len(self._entries), kb_path,
            "semantic" if _SEMANTIC_AVAILABLE else "tfidf",
        )

    def reload(self) -> None:
        """Reload the knowledge base from disk.

        Call this after editing ``knowledge_base.json`` to pick up changes
        immediately.  Also triggered automatically when ``auto_reload=True``
        and the file modification time changes.

        Equivalent to the  POST /reload/knowledge  API endpoint.
        """
        self._load_locked(self._config.knowledge_base_path)
        logger.info("Knowledge base reloaded (%d entries).", len(self._entries))

    def _check_and_reload(self) -> None:
        """Reload if the knowledge_base.json file has been modified."""
        try:
            mtime = self._config.knowledge_base_path.stat().st_mtime
        except OSError:
            return
        if mtime != self._kb_mtime and self._reload_lock.acquire(blocking=False):
            try:
                # Re-check inside lock to avoid double reload
                if self._config.knowledge_base_path.stat().st_mtime != self._kb_mtime:
                    logger.info("knowledge_base.json changed — hot-reloading …")
                    self._load_locked(self._config.knowledge_base_path)
            except Exception:
                logger.exception("Auto-reload failed — using previous index.")
            finally:
                self._reload_lock.release()

    # ─────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant knowledge chunks for *query*.

        The search is fully semantic when sentence-transformers is installed,
        or falls back to TF-IDF keyword matching otherwise.

        Parameters
        ----------
        query:
            The customer's question or utterance as plain text.
        top_k:
            Maximum number of chunks to return  (default: ``RAG_TOP_K`` env var, 3).
        threshold:
            Minimum similarity score to include a chunk  (default: ``RAG_SIMILARITY_THRESHOLD``, 0.3).

        Returns
        -------
        A list of ``RetrievedChunk`` objects sorted by relevance, highest first.

        Example
        -------
        ::

            chunks = engine.retrieve("How much does this cost?")
            # → [RetrievedChunk(category='pricing', question='How much does...', score=0.87), ...]
        """
        if not self._loaded:
            self.load()

        if self._auto_reload:
            self._check_and_reload()

        top_k     = top_k     if top_k     is not None else self._config.top_k
        threshold = threshold if threshold is not None else self._config.similarity_threshold

        if _SEMANTIC_AVAILABLE:
            scores = self._semantic_scores(query)
        else:
            scores = self._tfidf_scores(query)

        # Apply keyword bonus (works for both backends)
        query_lower = query.lower()
        for idx in range(len(scores)):
            bonus = sum(
                0.05 for kw in self._entries[idx]["keywords"] if kw in query_lower
            )
            scores[idx] += bonus

        # Collect entries above threshold, sort by score
        scored = [
            (scores[idx], idx)
            for idx in range(len(self._entries))
            if scores[idx] >= threshold
        ]
        scored.sort(reverse=True)

        results: list[RetrievedChunk] = []
        for score, idx in scored[:top_k]:
            entry = self._entries[idx]
            results.append(RetrievedChunk(
                category=entry["category"],
                question=entry["question"],
                answer=entry["answer"],
                score=round(float(score), 4),
            ))

        logger.debug(
            "RAG  query='%.60s'  hits=%d  backend=%s",
            query, len(results),
            "semantic" if _SEMANTIC_AVAILABLE else "tfidf",
        )
        return results

    # ─────────────────────────────────────────────────────────────────────
    # Context formatting  (for logging / debug)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a readable context block.

        The returned string is suitable for including in the LLM system
        prompt, log output, or the debug panel in tests.

        Example output::

            [KNOWLEDGE BASE CONTEXT]

            --- Reference 1 (pricing) ---
            Q: How much does final expense insurance cost per month?
            A: Monthly premiums vary by age, gender ...

            [END CONTEXT]
        """
        if not chunks:
            return ""
        lines = ["[KNOWLEDGE BASE CONTEXT]"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"\n--- Reference {i} ({chunk.category}) ---")
            lines.append(f"Q: {chunk.question}")
            lines.append(f"A: {chunk.answer}")
        lines.append("\n[END CONTEXT]")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # LLM context injection
    # ─────────────────────────────────────────────────────────────────────

    def inject_context(
        self,
        system_prompt: str,
        customer_query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """Retrieve relevant KB chunks and append them to *system_prompt*.

        This is the main integration point with the LLM pipeline.  Pass
        the result as the ``system`` message to whatever LLM you use.

        Call flow::

            customer says: "How much does this cost?"
                ↓
            retrieve pricing entries  (score ≥ threshold)
                ↓
            append [KNOWLEDGE BASE CONTEXT] block to system_prompt
                ↓
            LLM generates grounded response citing real pricing info

        Parameters
        ----------
        system_prompt:
            Your base system prompt (persona, instructions, etc.).
        customer_query:
            The customer's latest utterance — used as the retrieval query.
        top_k, threshold:
            Passed straight through to ``retrieve()``.

        Returns
        -------
        The system prompt with the KB context block appended.  If no
        relevant chunks are found the original *system_prompt* is returned
        unchanged.

        Example::

            prompt = engine.inject_context(
                "You are a final expense insurance specialist ...",
                "How much does a $15,000 policy cost at age 70?",
            )
            response = llm.chat(system=prompt, user=customer_query)
        """
        chunks = self.retrieve(customer_query, top_k=top_k, threshold=threshold)
        if not chunks:
            return system_prompt

        context_block = self.format_context(chunks)
        return (
            f"{system_prompt}\n\n"
            f"{context_block}\n\n"
            "Use the above context to answer the customer's question accurately. "
            "If the context does not contain the answer, rely on your general knowledge "
            "but stay consistent with the product details shown."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Semantic index (sentence-transformers)
    # ─────────────────────────────────────────────────────────────────────

    def _build_semantic_index(self) -> None:
        """Encode all KB entries with the sentence-transformers model.

        The model is loaded once and reused across reloads.
        Encoding ~50 entries takes ~200 ms on CPU.
        """
        if self._model is None:
            logger.info("Loading sentence-transformers model '%s' …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Model loaded.")

        # Embed  "question + answer"  concatenation for each entry.
        # Including the answer text means the entry will also match queries
        # that paraphrase the answer rather than the question.
        texts = [
            f"{e['question']} {e['answer']}"
            for e in self._entries
        ]
        self._entry_embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,      # unit vectors → dot product = cosine
        )

    def _semantic_scores(self, query: str) -> list[float]:
        """Return cosine similarity scores between *query* and every entry."""
        query_vec = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # Dot product of unit vectors == cosine similarity
        sims = (self._entry_embeddings @ query_vec).tolist()
        return sims

    # ─────────────────────────────────────────────────────────────────────
    # TF-IDF fallback index
    # ─────────────────────────────────────────────────────────────────────

    def _build_tfidf_index(self) -> None:
        """Build TF-IDF vectors for every KB entry (no extra dependencies)."""
        self._entry_vectors = []
        for entry in self._entries:
            tokens = self._tokenize(
                entry["question"] + " " + entry["answer"] + " " + " ".join(entry["keywords"])
            )
            self._entry_vectors.append(Counter(tokens))

        num_docs = len(self._entries)
        all_tokens: set[str] = set()
        for vec in self._entry_vectors:
            all_tokens.update(vec.keys())

        self._idf = {}
        for token in all_tokens:
            df = sum(1 for vec in self._entry_vectors if token in vec)
            self._idf[token] = math.log((num_docs + 1) / (df + 1)) + 1

    def _tfidf_scores(self, query: str) -> list[float]:
        """Return TF-IDF cosine similarity scores for *query*."""
        query_vec = Counter(self._tokenize(query))
        return [self._cosine_tfidf(query_vec, ev) for ev in self._entry_vectors]

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    _WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._WORD_RE.findall(text.lower())

    def _cosine_tfidf(self, vec_a: Counter, vec_b: Counter) -> float:
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(
            vec_a[t] * self._idf.get(t, 1) * vec_b[t] * self._idf.get(t, 1)
            for t in common
        )
        mag_a = math.sqrt(sum((vec_a[t] * self._idf.get(t, 1)) ** 2 for t in vec_a))
        mag_b = math.sqrt(sum((vec_b[t] * self._idf.get(t, 1)) ** 2 for t in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ─────────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        """Number of knowledge entries currently indexed."""
        return len(self._entries)

    @property
    def backend(self) -> str:
        """Which search backend is active: ``'semantic'`` or ``'tfidf'``."""
        return "semantic" if _SEMANTIC_AVAILABLE else "tfidf"

    @property
    def is_loaded(self) -> bool:
        return self._loaded
