import unittest
from app import app, distance, chunk_text, DIMS, db, DocumentDB


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_serves_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"My_personal_AI", response.data)

    def test_seed_vectors(self):
        response = self.client.get("/items")
        self.assertEqual(response.status_code, 200)
        items = response.get_json()
        self.assertEqual(len(items), 20)
        self.assertTrue(all(len(item["embedding"]) == DIMS for item in items))

    def test_search_algorithms(self):
        query = ",".join(map(str, db.items[1]["embedding"]))
        for algo in ("bruteforce", "kdtree", "hnsw"):
            response = self.client.get(
                "/search?v=" + query + "&k=3&metric=cosine&algo=" + algo
            )
            self.assertEqual(response.status_code, 200)
            results = response.get_json()["results"]
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0]["id"], 1)

    def test_invalid_vector(self):
        self.assertEqual(self.client.get("/search?v=1,2").status_code, 400)

    def test_hnsw_info(self):
        response = self.client.get("/hnsw-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["nodeCount"], 20)
        self.assertTrue(data["nodesPerLayer"])

    def test_distance_metrics(self):
        self.assertAlmostEqual(distance([1, 0], [1, 0], "cosine"), 0)
        self.assertAlmostEqual(distance([0, 0], [1, 1], "euclidean"), 2 ** 0.5)
        self.assertAlmostEqual(distance([1, 2], [3, 4], "manhattan"), 4)

    def test_document_embeddings_use_hnsw_retrieval(self):
        docs = DocumentDB()
        docs.insert("close match", "nearest document", [1.0, 0.0, 0.0])
        docs.insert("far match", "different document", [0.0, 1.0, 0.0])
        hits = docs.search([1.0, 0.0, 0.0], 1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1]["title"], "close match")

    def test_chunk_text(self):
        self.assertEqual(chunk_text("one two three"), ["one two three"])


if __name__ == "__main__":
    unittest.main()
