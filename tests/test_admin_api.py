import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.admin import router as admin_router


class AdminApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(admin_router)
        cls.client = TestClient(app)

    def test_admin_login_accepts_configured_password(self):
        response = self.client.post("/api/admin/login", json={"password": "admin123"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("token", response.json())

    def test_admin_login_rejects_bad_password(self):
        response = self.client.post("/api/admin/login", json={"password": "wrong-password"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Incorrect password")


if __name__ == "__main__":
    unittest.main()
