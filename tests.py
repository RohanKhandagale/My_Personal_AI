import unittest
from app import app, distance, chunk_text, DIMS, db


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

    def test_search(self):
        query = ",".join(map(str, db.items[1]["embedding"]))
        response = self.client.get("/search?v=" + query + "&k=3&metric=cosine&algo=bruteforce")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0]["id"], 1)

    def test_invalid_vector(self):
        self.assertEqual(self.client.get("/search?v=1,2").status_code, 400)

    def test_distance_metrics(self):
        self.assertAlmostEqual(distance([1, 0], [1, 0], "cosine"), 0)
        self.assertAlmostEqual(distance([0, 0], [1, 1], "euclidean"), 2 ** 0.5)
        self.assertAlmostEqual(distance([1, 2], [3, 4], "manhattan"), 4)

    def test_chunk_text(self):
        self.assertEqual(chunk_text("one two three"), ["one two three"])


if __name__ == "__main__":
    unittest.main()
