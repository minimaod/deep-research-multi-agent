"""Deterministic evidence deduplication and complete-envelope budget enforcement."""
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.harness.schemas import EvidenceItem
from app.harness.token_counter import counter


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return url
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return url
    if port and (parts.scheme, port) not in (("http", 80), ("https", 443)):
        host += f":{port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode(sorted(query)), ""))


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def content_fingerprint(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def features(text: str) -> list[str]:
    text = normalized_text(text)
    words = re.findall(r"[a-z0-9_.%]+|[\u4e00-\u9fff]", text)
    return words + [a + " " + b for a, b in zip(words, words[1:])]


def simhash64(text: str) -> int:
    vector = [0.0] * 64
    for word, freq in Counter(features(text)).items():
        value = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
        weight = 1 + math.log(freq)
        for bit in range(64):
            vector[bit] += weight if value & (1 << bit) else -weight
    return sum(1 << bit for bit, value in enumerate(vector) if value > 0)


def factual_signature(text: str) -> tuple:
    """Don't merge near-identical statements with changed numbers or negation."""
    text = normalized_text(text)
    return (tuple(re.findall(r"\d+(?:[.,]\d+)*(?:\s*(?:%|[a-z]+|[万亿年月日元美欧人民币公斤吨毫克]+))?", text)),
            tuple(re.findall(r"[$€£¥]", text)),
            tuple(re.findall(r"\b(?:not|no|never|without)\b|不|未|无|否", text)))


def relevance(text: str, query: str) -> float:
    wanted = set(features(query))
    return len(set(features(text)) & wanted) / max(1, len(wanted))


@dataclass
class TrimResult:
    content: str
    evidence: list[EvidenceItem]
    tokens_before: int
    tokens_after: int
    duplicates_removed: int
    fragments_dropped: int
    truncated: bool
    budget: int
    tokenizer: str

    def stats(self):
        return {k: v for k, v in vars(self).items() if k not in {"content", "evidence"}}


def render_evidence(items, metadata=None):
    return json.dumps({**(metadata or {}), "evidence": [x.model_dump(exclude_defaults=True) for x in items]}, ensure_ascii=False, separators=(",", ":"))


def trim_context(evidence: list[EvidenceItem], query: str, max_tokens: int = 1500,
                 metadata: dict | None = None, hamming_threshold: int = 3) -> TrimResult:
    if max_tokens < 32:
        raise ValueError("budget must be at least 32")
    metadata = {**(metadata or {}), "context_budget": max_tokens, "context_truncated": False}
    if counter.count(render_evidence([], metadata)) > max_tokens:
        raise ValueError("metadata alone exceeds context budget")
    before = counter.count(render_evidence(evidence, metadata))
    unique: list[EvidenceItem] = []
    hashes: list[tuple[str, int, tuple]] = []
    duplicates = 0
    for item in evidence:
        item = item.model_copy(update={"locator": normalize_url(item.locator)})
        exact, near, signature = content_fingerprint(item.text), simhash64(item.text), factual_signature(item.text)
        match = next((i for i, (fp, sh, sig) in enumerate(hashes)
                      if exact == fp or (len(item.text) >= 80 and signature == sig and (near ^ sh).bit_count() <= hamming_threshold)), None)
        if match is not None:
            old = unique[match]
            aliases = list(dict.fromkeys(old.aliases + [f"{item.source}|{item.id}|{item.locator}"]))
            unique[match] = old.model_copy(update={"aliases": aliases[:8]})
            duplicates += 1
        else:
            unique.append(item)
            hashes.append((exact, near, signature))
    # Prioritize matching conclusions, then cover each represented source.
    ranked = sorted(unique, key=lambda x: (relevance(x.text, query) + x.score * .15), reverse=True)
    ordered = []
    seen_sources = set()
    for item in ranked:
        if item.source not in seen_sources:
            ordered.append(item)
            seen_sources.add(item.source)
    ordered.extend(x for x in ranked if x not in ordered)
    selected = []
    dropped = 0
    for item in ordered:
        if counter.count(render_evidence(selected + [item], metadata)) <= max_tokens:
            selected.append(item)
            continue
        if item.kind == "row":
            dropped += 1
            continue
        # English periods only split before whitespace: decimals such as 20.5
        # must never become separate, selectable fragments.
        sentences = [s.strip() for s in re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+|\n+", item.text) if s.strip()]
        qualified = []
        for sentence in sentences:
            # Keep a following qualification together with its claim, e.g.
            # "Revenue is 42. However, this is not audited."
            if qualified and re.search(r"\b(?:however|but|unless|not|only)\b|但是|然而|仅限|未经|不代表", sentence, re.I):
                qualified[-1] += " " + sentence
            else:
                qualified.append(sentence)
        sentences = qualified
        prioritized = sorted(enumerate(sentences), key=lambda p: relevance(p[1], query), reverse=True)
        kept = []
        for index, sentence in prioritized:
            trial = sorted(kept + [(index, sentence)])
            piece = item.model_copy(update={"text": " ".join(s for _, s in trial)})
            if counter.count(render_evidence(selected + [piece], metadata)) <= max_tokens:
                kept = trial
        if kept:
            selected.append(item.model_copy(update={"text": " ".join(s for _, s in kept)}))
        dropped += 1
    metadata["context_truncated"] = dropped > 0
    content = render_evidence(selected, metadata)
    while selected and counter.count(content) > max_tokens:
        selected.pop()
        dropped += 1
        metadata["context_truncated"] = True
        content = render_evidence(selected, metadata)
    return TrimResult(content, selected, before, counter.count(content), duplicates, dropped,
                      dropped > 0, max_tokens, counter.name)
