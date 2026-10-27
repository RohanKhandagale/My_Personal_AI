"""Tests for the Python VectorDB port.  Run:  python -m unittest -v tests"""
import json
import threading
import unittest
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import app

Q = "0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1"


class FakeOllama(BaseHTTPRequestHandler):
    """Tiny stand-in for the Ollama REST API (bag-of-words embeddings)."""

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json({"models": []})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embeddings":
            v = [0.0] * 32
            for w in data["prompt"].lower().split():
                v[zlib.crc32(w.encode()) % 32] += 1
            self._json({"embedding": v})
        else:
            self._json({"response": "fake answer", "done": True})


def start(handler_or_server):
    t = threading.Thread(target=handler_or_server.serve_forever, daemon=True)
    t.start()


def call(base, path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode()


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.db = app.VectorDB(app.DIMS)
        app.load_demo(self.db)
        self.q = app.parse_vec(Q)

    def test_demo_loaded(self):
        self.assertEqual(self.db.size(), 20)

    def test_all_algorithms_agree_with_bruteforce(self):
        for metric in ("cosine", "euclidean", "manhattan"):
            ref = [h["id"] for h in self.db.search(self.q, 5, metric, "bruteforce")["hits"]]
            kd = [h["id"] for h in self.db.search(self.q, 5, metric, "kdtree")["hits"]]
            self.assertEqual(ref, kd, metric)
        ref = [h["id"] for h in self.db.search(self.q, 5, "cosine", "bruteforce")["hits"]]
        hn = [h["id"] for h in self.db.search(self.q, 5, "cosine", "hnsw")["hits"]]
        self.assertEqual(ref, hn)

    def test_remove_and_hnsw_info(self):
        info = self.db.hnsw_info()
        self.assertEqual(info["nodeCount"], 20)
        self.assertTrue(self.db.remove(1))
        self.assertFalse(self.db.remove(1))
        self.assertEqual(self.db.size(), 19)
        ids = [h["id"] for h in self.db.search(self.q, 19, "cosine", "kdtree")["hits"]]
        self.assertNotIn(1, ids)

    def test_mt19937_matches_cpp(self):
        # std::mt19937 default-seeded with 42: first output is 1608637542
        self.assertEqual(app.MT19937(42).next_u32(), 1608637542)

    def test_chunker(self):
        self.assertEqual(app.chunk_text(""), [])
        self.assertEqual(app.chunk_text("a b c"), ["a b c"])
        chunks = app.chunk_text(" ".join(str(i) for i in range(600)))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0].split()), 250)
        self.assertEqual(chunks[1].split()[0], "220")  # 250 - 30 overlap

    def test_json_helpers(self):
        self.assertEqual(app.j_s('a"b\\c\n'), '"a\\"b\\\\c\\n"')
        self.assertEqual(app.j_vec([0.9, 0.12345]), "[0.9000,0.1235]")
        self.assertEqual(app.parse_vec("1, 2,x,3.5"), [1.0, 2.0, 3.5])


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.db = app.VectorDB(app.DIMS)
        app.doc_db = app.DocumentDB()
        app.load_demo(app.db)
        cls.fake = HTTPServer(("127.0.0.1", 0), FakeOllama)
        start(cls.fake)
        app.ollama = app.OllamaClient("127.0.0.1", cls.fake.server_address[1])
        cls.srv = app.make_server("127.0.0.1", 0)
        start(cls.srv)
        cls.base = "http://127.0.0.1:%d" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.fake.shutdown()

    def test_ui_served(self):
        status, body = call(self.base, "/")
        self.assertEqual(status, 200)
        self.assertIn("<html", body.lower())

    def test_search_shape(self):
        status, body = call(self.base, "/search?v=%s&k=3&metric=cosine&algo=hnsw" % Q)
        data = json.loads(body)
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["algo"], "hnsw")
        self.assertEqual(set(data["results"][0]), {"id", "metadata", "category", "distance", "embedding"})
        self.assertIn("latencyUs", data)

    def test_search_bad_vector(self):
        _, body = call(self.base, "/search?v=1,2&k=3")
        self.assertEqual(body, '{"error":"need 16D vector"}')

    def test_stats_and_benchmark(self):
        _, body = call(self.base, "/stats")
        self.assertEqual(json.loads(body)["dims"], 16)
        _, body = call(self.base, "/benchmark?v=%s&k=5&metric=cosine" % Q)
        self.assertEqual(set(json.loads(body)), {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"})

    def test_insert_delete(self):
        emb = [0.5] * 8 + [0.1] * 8
        _, body = call(self.base, "/insert", "POST", {"metadata": "x", "category": "c", "embedding": emb})
        new_id = json.loads(body)["id"]
        _, body = call(self.base, "/delete/%d" % new_id, "DELETE")
        self.assertEqual(body, '{"ok":true}')
        _, body = call(self.base, "/delete/%d" % new_id, "DELETE")
        self.assertEqual(body, '{"ok":false}')
        _, body = call(self.base, "/insert", "POST", {"metadata": "x", "embedding": [1, 2]})
        self.assertEqual(body, '{"error":"invalid body"}')

    def test_hnsw_info(self):
        _, body = call(self.base, "/hnsw-info")
        info = json.loads(body)
        self.assertEqual(info["nodeCount"], len(info["nodes"]))

    def test_rag_pipeline(self):
        _, body = call(self.base, "/doc/insert", "POST",
                       {"title": "Sorting", "text": "quicksort mergesort heapsort sorting"})
        self.assertEqual(json.loads(body)["chunks"], 1)
        call(self.base, "/doc/insert", "POST", {"title": "Cooking", "text": "pasta tomato basil"})
        _, body = call(self.base, "/doc/list")
        self.assertEqual(len(json.loads(body)), 2)
        _, body = call(self.base, "/doc/ask", "POST", {"question": "quicksort sorting", "k": 1})
        data = json.loads(body)
        self.assertEqual(data["answer"], "fake answer")
        self.assertEqual(data["contexts"][0]["title"], "Sorting")
        _, body = call(self.base, "/doc/search", "POST", {"question": "pasta", "k": 1})
        self.assertEqual(json.loads(body)["contexts"][0]["title"], "Cooking")
        _, body = call(self.base, "/status")
        self.assertTrue(json.loads(body)["ollamaAvailable"])


class OfflineTests(unittest.TestCase):
    def test_ollama_offline_errors(self):
        client = app.OllamaClient("127.0.0.1", 1)  # nothing listens here
        self.assertFalse(client.is_available())
        self.assertEqual(client.embed("hi"), [])
        self.assertEqual(client.generate("hi"), "ERROR: Ollama unavailable. Run: ollama serve")


if __name__ == "__main__":
    unittest.main()
