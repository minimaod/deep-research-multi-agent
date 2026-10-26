"""No downloads on the execution path. Counts are explicitly proxy counts."""
import base64
import hashlib
import os
import tempfile
import json
import logging
from pathlib import Path


class TokenCounter:
    """Use an already cached cl100k vocabulary, otherwise a conservative byte budget.

    Neither is the API billing counter for arbitrary OpenAI-compatible models.
    The fallback is an upper bound for byte-level BPE, not a universal tokenizer.
    """
    def __init__(self):
        self.encoding = None
        self.name = "utf8-byte-budget-v1 (proxy, conservative)"
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        cache = Path(os.getenv("TIKTOKEN_CACHE_DIR", os.getenv("DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache"))))
        vocab = cache / hashlib.sha1(url.encode()).hexdigest()
        if vocab.is_file():
            try:
                import ast
                import inspect
                import re
                import tiktoken
                from tiktoken_ext.openai_public import cl100k_base
                # Do not call cl100k_base(): that function can download its vocabulary.
                pattern = re.search(r'pat_str=(r?".*?")', inspect.getsource(cl100k_base))
                raw = vocab.read_bytes()
                expected = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
                if hashlib.sha256(raw).hexdigest() != expected or not pattern:
                    raise ValueError("invalid local cl100k cache")
                ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in raw.splitlines())}
                self.encoding = tiktoken.Encoding(name="local-cl100k-proxy", pat_str=ast.literal_eval(pattern[1]), mergeable_ranks=ranks, special_tokens={})
                self.name = "local-cl100k-proxy (not provider usage)"
            except (ImportError, ValueError, OSError):
                logging.getLogger(__name__).warning("Local tokenizer unavailable; using offline byte budget")

    def count(self, text: str) -> int:
        if self.encoding is not None:
            return len(self.encoding.encode(text, disallowed_special=()))
        return len(text.encode("utf-8"))


counter = TokenCounter()


def count_model_context(messages, system="", tools=None):
    """Deterministic serialized-input proxy, not provider billing token usage."""
    wire = []
    for message in messages:
        if isinstance(message, dict):
            wire.append(message)
        else:
            item = {"role": message.type, "content": message.content}
            if getattr(message, "tool_calls", None):
                item["tool_calls"] = message.tool_calls
            wire.append(item)
    return counter.count(json.dumps({"system": system, "messages": wire, "tools": tools or []}, ensure_ascii=False, separators=(",", ":")))
