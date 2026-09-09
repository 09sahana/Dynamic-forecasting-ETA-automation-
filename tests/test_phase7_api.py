"""
tests/test_phase7_api.py
=========================
Integration tests for Phase 7 FastAPI endpoints and journey aggregation.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from api.app import app


class TestPhase7API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model_version"], "v6")

    def test_model_metrics_endpoint(self):
        resp = self.client.get("/model/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["version"], "v6")
        self.assertIn("mae_non_capped", data)
        self.assertAlmostEqual(data["mae_non_capped"], 24.90, places=2)
        self.assertIn("leakage_audit_summary", data)
        self.assertIn("features", data)
        self.assertGreater(len(data["features"]), 15)

    def test_predict_endpoint_nominal(self):
        payload = {
            "train_no": 12952,
            "date": "2026-09-09",
            "station": "ST",
            "previous_station_delay": 15.0,
            "temperature_c": 32.0,
            "precipitation_mm": 0.0,
            "visibility_m": 8000.0,
            "event_elapsed_min": 0.0,
            "event_type": "NONE",
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Validate Chapter 8 schema
        self.assertEqual(data["train_no"], 12952)
        self.assertEqual(data["station_code"], "ST")
        self.assertIn("predicted_delay_min", data)
        self.assertIsNotNone(data["predicted_delay_min"])
        self.assertIn("predicted_delay_range_min", data)
        self.assertEqual(len(data["predicted_delay_range_min"]), 2)
        self.assertIn("predicted_eta", data)
        self.assertIn("confidence", data)
        self.assertIn(data["confidence"], ("High", "Medium", "Low"))
        self.assertIn("severe_delay_risk", data)
        self.assertIn("chronic_delay_tier", data)
        self.assertIn("explanations", data)
        self.assertGreaterEqual(len(data["explanations"]), 1)

    def test_predict_endpoint_chronic_train(self):
        # Train 12508 is known chronic tier T5
        payload = {
            "train_no": 12508,
            "date": "2026-09-09",
            "station": "KOTA",
            "previous_station_delay": 45.0,
            "event_type": "NONE",
            "event_elapsed_min": 0.0,
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["chronic_delay_tier"], "T5_worst")

    def test_full_journey_eta_chaining(self):
        resp = self.client.get("/train/12952/eta?current_delay=16.0")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["train_no"], 12952)
        self.assertIn("stations", data)
        self.assertGreater(len(data["stations"]), 3)
        self.assertEqual(data["origin_station"], "NDLS")
        self.assertEqual(data["destination_station"], "BCT")

        # Verify progressive delays and ETAs
        for stop in data["stations"]:
            self.assertIn("station_code", stop)
            self.assertIn("scheduled_arrival", stop)
            self.assertIn("predicted_eta", stop)

        # Final destination ETA should be computed
        self.assertIsNotNone(data["final_predicted_eta"])
        self.assertIsNotNone(data["final_predicted_delay_min"])

    def test_stream_status_and_mode_switch(self):
        resp = self.client.get("/api/stream/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("mode", data)

        # Switch to railradar
        switch_resp = self.client.post("/api/stream/mode?mode=railradar&train_no=12952")
        self.assertEqual(switch_resp.status_code, 200)
        self.assertEqual(switch_resp.json()["mode"], "railradar")

        # Switch back to simulator
        switch_resp2 = self.client.post("/api/stream/mode?mode=simulator&train_no=12952")
        self.assertEqual(switch_resp2.status_code, 200)
        self.assertEqual(switch_resp2.json()["mode"], "simulator")


if __name__ == "__main__":
    unittest.main()
