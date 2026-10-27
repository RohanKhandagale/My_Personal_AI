#!/usr/bin/env python3
"""VectorDB - a vector database built from scratch, with a web UI and a RAG pipeline.

Python port of the original C++ ``main.cpp``.  It implements three search
algorithms side by side (HNSW, KD-Tree, Brute Force), three distance metrics
(cosine, euclidean, manhattan), 16-D demo vectors, real document embeddings via
Ollama, and a RAG pipeline.  It serves the same REST API and the same
``index.html`` UI on http://localhost:8080 as the C++ server did.

Only the Python standard library is used (``http.server`` replaces
cpp-httplib, ``urllib`` replaces httplib's client).

Run:   python app.py
Open:  http://localhost:8080
"""
from __future__ import annotations

import heapq
import json
import math
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

DIMS = 16  # demo vectors
# Doc embeddings dimension is determined at runtime from Ollama's model output

ROOT = Path(__file__).resolve().parent
HOST = os.getenv("VECTORDB_HOST", "0.0.0.0")
PORT = int(os.getenv("VECTORDB_PORT", "8080"))


# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]
    norm: float = field(default=0.0, repr=False)  # cached |emb| (speeds up cosine)

    def __post_init__(self) -> None:
        self.norm = math.sqrt(_dot(self.emb, self.emb))


DistFn = Callable[[List[float], List[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

_sumprod = getattr(math, "sumprod", None)  # Python 3.12+: fast C dot product


def _dot(a: List[float], b: List[float]) -> float:
    if _sumprod is not None:
        return _sumprod(a, b)
    return sum(x * y for x, y in zip(a, b))


def euclidean(a: List[float], b: List[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def cosine(a: List[float], b: List[float]) -> float:
    dot = _dot(a, b)
    na = _dot(a, a)
    nb = _dot(b, b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        r = [(dist(q, v.emb), v.id) for v in self.items]
        r.sort()
        return r[:k] if len(r) > k else r

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _ins(self, n: Optional[KDNode], v: VectorItem, d: int) -> KDNode:
        # iterative version of the recursive C++ insert (same tree shape)
        if n is None:
            return KDNode(v)
        cur = n
        depth = d
        while True:
            ax = depth % self.dims
            if v.emb[ax] < cur.item.emb[ax]:
                if cur.left is None:
                    cur.left = KDNode(v)
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = KDNode(v)
                    break
                cur = cur.right
            depth += 1
        return n

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def _knn(self, n: Optional[KDNode], q: List[float], k: int, d: int,
             dist: DistFn, heap: list) -> None:
        # heap is a max-heap of (dist, id) pairs, stored negated for heapq
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, -n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer, farther = (n.left, n.right) if diff < 0 else (n.right, n.left)
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        r = [(-nd, -nid) for nd, nid in heap]
        r.sort()
        return r

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW - Hierarchical Navigable Small World
# =====================================================================

class MT19937:
    """Mersenne Twister identical to C++ std::mt19937 (so HNSW levels match)."""

    def __init__(self, seed: int) -> None:
        self.mt = [0] * 624
        self.idx = 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            prev = self.mt[i - 1]
            self.mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF

    def _twist(self) -> None:
        mt = self.mt
        for i in range(624):
            y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
            v = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908B0DF
            mt[i] = v
        self.idx = 0

    def next_u32(self) -> int:
        if self.idx >= 624:
            self._twist()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def _f32(x: float) -> float:
    """Round a Python float to float32 precision."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


class HNSW:
    class Node:
        __slots__ = ("item", "max_lyr", "nbrs")

        def __init__(self, item: VectorItem, max_lyr: int) -> None:
            self.item = item
            self.max_lyr = max_lyr
            self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]

    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.G: Dict[int, "HNSW.Node"] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = _f32(1.0 / _f32(math.log(m)))
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = MT19937(42)

    def _rand_level(self) -> int:
        # std::uniform_real_distribution<float>(0,1) == generate_canonical<float,24>
        u = _f32(float(self.rng.next_u32()) / 4294967296.0)
        if u >= 1.0:
            u = 1.0 - 2.0 ** -24
        if u <= 0.0:
            u = 2.0 ** -32  # avoid log(0); probability 2^-32
        return int(math.floor(_f32(-_f32(math.log(u)) * self.mL)))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int,
                      dist: DistFn) -> List[Tuple[float, int]]:
        G = self.G
        vis = {ep}
        cands: list = []   # min-heap of (dist, id)
        found: list = []   # max-heap of (dist, id), stored negated

        d0 = dist(q, G[ep].item.emb)
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found, (-d0, -ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = G[cid]
            if lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in G:
                    continue
                vis.add(nid)
                nd = dist(q, G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, -nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        res = [(-nd, -nid) for nd, nid in found]
        res.sort()
        return res

    @staticmethod
    def _select_nbrs(cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        G = self.G
        id_ = item.id
        lvl = self._rand_level()
        G[id_] = HNSW.Node(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in G:
                    continue
                nnode = G[nid]
                while len(nnode.nbrs) <= lc:
                    nnode.nbrs.append([])
                conn = nnode.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = [(dist(nnode.item.emb, G[c].item.emb), c) for c in conn if c in G]
                    ds.sort()
                    nnode.nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]
        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        G = self.G
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        if len(W) > k:
            W = W[:k]
        return W

    def remove(self, id_: int) -> None:
        G = self.G
        if id_ not in G:
            return
        for node in G.values():
            for i, layer in enumerate(node.nbrs):
                if id_ in layer:
                    node.nbrs[i] = [x for x in layer if x != id_]
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del G[id_]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes, edges = [], []
        for id_, nd in self.G.items():
            nodes.append({"id": id_, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_lyr})
            lc = 0
            while lc <= nd.max_lyr and lc < max_l:
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
                lc += 1
        return {"topLayer": self.top_layer, "nodeCount": len(self.G),
                "nodesPerLayer": nodes_per_layer, "edgesPerLayer": edges_per_layer,
                "nodes": nodes, "edges": edges}

    def size(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

def _now_us() -> int:
    return time.perf_counter_ns() // 1000


class VectorDB:
    def __init__(self, d: int) -> None:
        self.dims = d
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(d)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, list(emb))
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = _now_us()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = _now_us() - t0
            hits = []
            for d, id_ in raw:
                v = self.store.get(id_)
                if v is not None:
                    hits.append({"id": id_, "meta": v.metadata, "cat": v.category,
                                 "emb": v.emb, "dist": d})
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)

            def timed(fn) -> int:
                t = _now_us()
                fn()
                return _now_us() - t

            return {
                "bfUs": timed(lambda: self.bf.knn(q, k, dfn)),
                "kdUs": timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n": len(self.store),
            }

    def all(self) -> List[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# =====================================================================
#  JSON HELPERS  (output formatting identical to the C++ server)
# =====================================================================

def j_s(s: str) -> str:
    """Quote a string exactly like the C++ jS(): only " \\ \\n \\r \\t are escaped."""
    o = ['"']
    for c in s:
        if c == '"':
            o.append('\\"')
        elif c == "\\":
            o.append("\\\\")
        elif c == "\n":
            o.append("\\n")
        elif c == "\r":
            o.append("\\r")
        elif c == "\t":
            o.append("\\t")
        else:
            o.append(c)
    o.append('"')
    return "".join(o)


def j_vec(v: List[float]) -> str:
    return "[" + ",".join(f"{x:.4f}" for x in v) + "]"


_FLOAT_PREFIX = re.compile(r"\s*[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)",
                           re.IGNORECASE)


def stof(t: str) -> float:
    """std::stof: parse a leading float, raise ValueError when there is none."""
    m = _FLOAT_PREFIX.match(t)
    if not m:
        raise ValueError(t)
    return float(m.group(0))


def stoi(t: str) -> int:
    """std::stoi: parse a leading integer, raise ValueError when there is none."""
    m = re.match(r"\s*[+-]?\d+", t)
    if not m:
        raise ValueError(t)
    return int(m.group(0))


def parse_vec(s: str) -> List[float]:
    v: List[float] = []
    for t in s.split(","):
        try:
            v.append(stof(t))
        except ValueError:
            pass
    return v


def extract_str(body: str, key: str) -> str:
    """Extract a JSON string field value (handles basic escape sequences)."""
    p = body.find('"' + key + '"')
    if p == -1:
        return ""
    p = body.find(":", p) + 1
    while p < len(body) and body[p] in " \t":
        p += 1
    if p >= len(body) or body[p] != '"':
        return ""
    p += 1
    out = []
    while p < len(body):
        c = body[p]
        if c == '"':
            break
        if c == "\\" and p + 1 < len(body):
            p += 1
            c = body[p]
            out.append({"n": "\n", "r": "\r", "t": "\t"}.get(c, c))
        else:
            out.append(c)
        p += 1
    return "".join(out)


def extract_int(body: str, key: str, default: int = 0) -> int:
    p = body.find('"' + key + '"')
    if p == -1:
        return default
    p = body.find(":", p) + 1
    while p < len(body) and body[p] in " \t":
        p += 1
    try:
        return stoi(body[p:])
    except ValueError:
        return default


def _json_obj(body: str) -> Optional[dict]:
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def body_str(body: str, key: str) -> str:
    """Prefer a real JSON parse (handles \\uXXXX); fall back to the C++ scanner."""
    obj = _json_obj(body)
    if obj is not None:
        val = obj.get(key)
        return val if isinstance(val, str) else ""
    return extract_str(body, key)


def body_int(body: str, key: str, default: int = 0) -> int:
    obj = _json_obj(body)
    if obj is not None:
        val = obj.get(key)
        return val if isinstance(val, int) and not isinstance(val, bool) else default
    return extract_int(body, key, default)


def parse_body(b: str) -> Tuple[bool, str, str, List[float]]:
    meta = body_str(b, "metadata")
    cat = body_str(b, "category")
    emb: List[float] = []
    obj = _json_obj(b)
    if obj is not None and isinstance(obj.get("embedding"), list):
        emb = [float(x) for x in obj["embedding"] if isinstance(x, (int, float))]
    else:
        p = b.find('"embedding"')
        if p != -1:
            p = b.find("[", p)
            if p != -1:
                e = b.find("]", p)
                if e != -1:
                    emb = parse_vec(b[p + 1:e])
    return (bool(meta) and bool(emb)), meta, cat, emb


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks: List[str] = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# =====================================================================
#  OLLAMA CLIENT  - wraps local Ollama REST API
#  Install:  https://ollama.com
#  Models:   ollama pull nomic-embed-text
#            ollama pull llama3.2
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.base = f"http://{host}:{port}"
        self.embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.gen_model = os.getenv("OLLAMA_GEN_MODEL", "llama3.2")

    def _post(self, path: str, payload: dict, timeout: float) -> Optional[dict]:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def embed(self, text: str) -> List[float]:
        """Returns an empty list if Ollama is not running or the model is missing."""
        data = self._post("/api/embeddings", {"model": self.embed_model, "prompt": text}, 30)
        if not data or not isinstance(data.get("embedding"), list):
            return []
        return [float(x) for x in data["embedding"]]

    def generate(self, prompt: str) -> str:
        """Returns an error string if Ollama is unavailable."""
        data = self._post("/api/generate",
                          {"model": self.gen_model, "prompt": prompt, "stream": False}, 180)
        if data is None:
            return "ERROR: Ollama unavailable. Run: ollama serve"
        resp = data.get("response")
        return resp if isinstance(resp, str) else ""


# =====================================================================
#  DOCUMENT DATABASE  - HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    def __init__(self) -> None:
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()  # brute force fallback for small sets
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0  # determined from first inserted embedding

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        """Insert one chunk with its pre-computed embedding."""
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        """Semantic search - returns top-k most similar chunks."""
        with self.mu:
            if not self.store:
                return []
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
            return [(d, self.store[i]) for d, i in raw if i in self.store and d <= max_dist]

    def remove(self, id_: int) -> bool:
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

DEMO_DATA = [
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    ("Linked List: nodes connected by pointers", "cs",
     [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
]


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_DATA:
        db.insert(meta, cat, [_f32(x) for x in emb], dist)


# =====================================================================
#  HTTP SERVER
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

ERR_NEED_VEC = '{"error":"need ' + str(DIMS) + 'D vector"}'
ERR_OLLAMA_INSERT = ('{"error":"Ollama unavailable. Install from https://ollama.com then run: '
                     'ollama pull nomic-embed-text && ollama pull llama3.2"}')


def _json_bool(b: bool) -> str:
    return "true" if b else "false"


def route_search(q: dict) -> str:
    v = parse_vec(_param(q, "v"))
    if len(v) != DIMS:
        return ERR_NEED_VEC
    try:
        k = stoi(_param(q, "k"))
    except ValueError:
        k = 5
    metric = _param(q, "metric") or "cosine"
    algo = _param(q, "algo") or "hnsw"
    out = db.search(v, k, metric, algo)
    parts = []
    for h in out["hits"]:
        parts.append('{"id":%d,"metadata":%s,"category":%s,"distance":%.6f,"embedding":%s}'
                     % (h["id"], j_s(h["meta"]), j_s(h["cat"]), h["dist"], j_vec(h["emb"])))
    return '{"results":[%s],"latencyUs":%d,"algo":%s,"metric":%s}' % (
        ",".join(parts), out["us"], j_s(out["algo"]), j_s(out["metric"]))


def route_insert(body: str) -> str:
    ok, meta, cat, emb = parse_body(body)
    if not ok or len(emb) != DIMS:
        return '{"error":"invalid body"}'
    id_ = db.insert(meta, cat, [_f32(x) for x in emb], get_dist_fn("cosine"))
    return '{"id":%d}' % id_


def route_items() -> str:
    return "[" + ",".join(
        '{"id":%d,"metadata":%s,"category":%s,"embedding":%s}'
        % (v.id, j_s(v.metadata), j_s(v.category), j_vec(v.emb)) for v in db.all()) + "]"


def route_benchmark(q: dict) -> str:
    v = parse_vec(_param(q, "v"))
    if len(v) != DIMS:
        return ERR_NEED_VEC
    try:
        k = stoi(_param(q, "k"))
    except ValueError:
        k = 5
    metric = _param(q, "metric") or "cosine"
    b = db.benchmark(v, k, metric)
    return '{"bruteforceUs":%d,"kdtreeUs":%d,"hnswUs":%d,"itemCount":%d}' % (
        b["bfUs"], b["kdUs"], b["hnswUs"], b["n"])


def route_hnsw_info() -> str:
    gi = db.hnsw_info()
    nodes = ",".join('{"id":%d,"metadata":%s,"category":%s,"maxLyr":%d}'
                     % (n["id"], j_s(n["metadata"]), j_s(n["category"]), n["maxLyr"])
                     for n in gi["nodes"])
    edges = ",".join('{"src":%d,"dst":%d,"lyr":%d}' % (e["src"], e["dst"], e["lyr"])
                     for e in gi["edges"])
    return '{"topLayer":%d,"nodeCount":%d,"nodesPerLayer":[%s],"edgesPerLayer":[%s],"nodes":[%s],"edges":[%s]}' % (
        gi["topLayer"], gi["nodeCount"],
        ",".join(map(str, gi["nodesPerLayer"])), ",".join(map(str, gi["edgesPerLayer"])),
        nodes, edges)


def route_doc_insert(body: str) -> str:
    """Chunks the text, embeds each chunk via Ollama, stores in DocumentDB."""
    title = body_str(body, "title")
    text = body_str(body, "text")
    if not title or not text:
        return '{"error":"need title and text"}'
    chunks = chunk_text(text, 250, 30)
    ids: List[int] = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return ERR_OLLAMA_INSERT
        chunk_title = "%s [%d/%d]" % (title, i + 1, len(chunks)) if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))
    return '{"ids":[%s],"chunks":%d,"dims":%d}' % (",".join(map(str, ids)), len(chunks), doc_db.get_dims())


def route_doc_list() -> str:
    parts = []
    for d in doc_db.all():
        preview = d.text[:120]
        if len(d.text) > 120:
            preview += "\u2026"
        parts.append('{"id":%d,"title":%s,"preview":%s,"words":%d}'
                     % (d.id, j_s(d.title), j_s(preview), d.text.count(" ") + 1))
    return "[" + ",".join(parts) + "]"


def route_doc_search(body: str) -> str:
    """Fast retrieval for the UI visualizer."""
    question = body_str(body, "question")
    k = body_int(body, "k", 3)
    if not question:
        return '{"error":"need question"}'
    q_emb = ollama.embed(question)
    if not q_emb:
        return '{"error":"Ollama unavailable"}'
    hits = doc_db.search(q_emb, k)
    ctx = ",".join('{"id":%d,"title":%s,"distance":%.4f}' % (d.id, j_s(d.title), dist)
                   for dist, d in hits)
    return '{"contexts":[%s]}' % ctx


def route_doc_ask(body: str) -> str:
    """Full RAG pipeline: embed -> retrieve -> generate."""
    question = body_str(body, "question")
    k = body_int(body, "k", 3)
    if not question:
        return '{"error":"need question"}'

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return '{"error":"Ollama unavailable"}'

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx = "".join("[%d] %s:\n%s\n\n" % (i + 1, d.title, d.text) for i, (_, d) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        "Context:\n" + ctx +
        "Question: " + question + "\n\n"
        "Answer:")

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    contexts = ",".join('{"id":%d,"title":%s,"text":%s,"distance":%.4f}'
                        % (d.id, j_s(d.title), j_s(d.text), dist) for dist, d in hits)
    return '{"answer":%s,"model":%s,"contexts":[%s],"docCount":%d}' % (
        j_s(answer), j_s(ollama.gen_model), contexts, doc_db.size())


def route_status() -> str:
    up = ollama.is_available()
    return ('{"ollamaAvailable":%s,"embedModel":%s,"genModel":%s,"docCount":%d,'
            '"docDims":%d,"demoDims":%d,"demoCount":%d}') % (
        _json_bool(up), j_s(ollama.embed_model), j_s(ollama.gen_model),
        doc_db.size(), doc_db.get_dims(), DIMS, db.size())


def route_stats() -> str:
    return ('{"count":%d,"dims":%d,"algorithms":["bruteforce","kdtree","hnsw"],'
            '"metrics":["euclidean","cosine","manhattan"]}') % (db.size(), DIMS)


def _param(q: dict, name: str) -> str:
    vals = q.get(name)
    return vals[0] if vals else ""


class Handler(BaseHTTPRequestHandler):
    server_version = "VectorDB/1.0"
    protocol_version = "HTTP/1.1"

    # ---- helpers ------------------------------------------------------
    def _send(self, status: int, body: str = "", ctype: str = "application/json",
              cors: bool = True) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if data or status != 204:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length).decode("utf-8", errors="replace") if length else ""

    def log_message(self, fmt, *args) -> None:  # keep the console quiet like the C++ server
        pass

    # ---- verbs --------------------------------------------------------
    def do_OPTIONS(self) -> None:
        self._send(204)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        q = parse_qs(url.query, keep_blank_values=True)
        path = url.path
        try:
            if path == "/search":
                self._send(200, route_search(q))
            elif path == "/items":
                self._send(200, route_items())
            elif path == "/benchmark":
                self._send(200, route_benchmark(q))
            elif path == "/hnsw-info":
                self._send(200, route_hnsw_info())
            elif path == "/doc/list":
                self._send(200, route_doc_list())
            elif path == "/status":
                self._send(200, route_status())
            elif path == "/stats":
                self._send(200, route_stats())
            elif path == "/":
                f = ROOT / "index.html"
                if not f.is_file():
                    self._send(404, cors=False)
                else:
                    self._send(200, f.read_text(encoding="utf-8"), "text/html; charset=utf-8", cors=False)
            else:
                self._send(404, cors=False)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        body = self._read_body()
        try:
            if path == "/insert":
                self._send(200, route_insert(body))
            elif path == "/doc/insert":
                self._send(200, route_doc_insert(body))
            elif path == "/doc/search":
                self._send(200, route_doc_search(body))
            elif path == "/doc/ask":
                self._send(200, route_doc_ask(body))
            else:
                self._send(404, cors=False)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_DELETE(self) -> None:
        path = urlsplit(self.path).path
        m = re.fullmatch(r"/delete/(\d+)", path)
        d = re.fullmatch(r"/doc/delete/(\d+)", path)
        try:
            if m:
                self._send(200, '{"ok":%s}' % _json_bool(db.remove(int(m.group(1)))))
            elif d:
                self._send(200, '{"ok":%s}' % _json_bool(doc_db.remove(int(d.group(1)))))
            else:
                self._send(404, cors=False)
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> int:
    load_demo(db)

    # Check Ollama at startup (non-fatal)
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===", flush=True)
    print(f"http://localhost:{PORT}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama: " + ("ONLINE" if ollama_up else "OFFLINE (install from ollama.com)"))
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    sys.stdout.flush()

    server = make_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
