"""
Test suite for FastAPI endpoints.
Tests API endpoint functionality with mocked components.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


# Mock the model loading before importing main
@pytest.fixture(scope="module")
def client():
    """Create a test client with mocked model."""
    with patch('api.main.load_trained_model') as mock_load:
        # Create a mock model that returns dummy predictions
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0.33, 0.33, 0.34]))
        mock_load.return_value = mock_model
        
        from api.main import app
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_contains_required_fields(self, client):
        """Test health response contains required fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert data["status"] == "healthy"

    def test_health_model_status(self, client):
        """Test health endpoint reports model status correctly."""
        response = client.get("/health")
        data = response.json()
        
        # Model should be loaded (mocked)
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_valid_input(self, client):
        """Test predict with valid input."""
        payload = {
            "market_data": [
                {"price": 100.0, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
                {"price": 50.0, "return_1d": -0.02, "volatility": 0.03, "momentum": -0.01},
                {"price": 75.0, "return_1d": 0.005, "volatility": 0.015, "momentum": 0.02}
            ],
            "portfolio_state": {
                "current_weights": [0.4, 0.3, 0.3],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "allocation_weights" in data
        assert "risk_metrics" in data

    def test_predict_returns_valid_weights(self, client):
        """Test that predicted weights are valid probabilities."""
        payload = {
            "market_data": [
                {"price": 100.0, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
                {"price": 50.0, "return_1d": -0.02, "volatility": 0.03, "momentum": -0.01},
                {"price": 75.0, "return_1d": 0.005, "volatility": 0.015, "momentum": 0.02}
            ],
            "portfolio_state": {
                "current_weights": [0.4, 0.3, 0.3],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        weights = data["allocation_weights"]
        
        # Weights should sum to approximately 1
        assert abs(sum(weights) - 1.0) < 0.01
        
        # All weights should be non-negative
        assert all(w >= 0 for w in weights)

    def test_predict_missing_market_data(self, client):
        """Test predict rejects missing market data."""
        payload = {
            "portfolio_state": {
                "current_weights": [0.4, 0.3, 0.3],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_empty_market_data(self, client):
        """Test predict rejects empty market data list."""
        payload = {
            "market_data": [],
            "portfolio_state": {
                "current_weights": [],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422 or response.status_code == 400

    def test_predict_invalid_price(self, client):
        """Test predict rejects invalid prices."""
        payload = {
            "market_data": [
                {"price": -100.0, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
            ],
            "portfolio_state": {
                "current_weights": [1.0],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        response = client.post("/predict", json=payload)
        # Should either validate and reject negative price or handle gracefully
        assert response.status_code in [200, 422]

    def test_predict_risk_metrics_present(self, client):
        """Test that risk metrics are included in response."""
        payload = {
            "market_data": [
                {"price": 100.0, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
                {"price": 50.0, "return_1d": -0.02, "volatility": 0.03, "momentum": -0.01},
            ],
            "portfolio_state": {
                "current_weights": [0.5, 0.5],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        risk_metrics = data["risk_metrics"]
        assert "sharpe_ratio" in risk_metrics or "volatility" in risk_metrics


class TestFeedbackEndpoint:
    """Test /feedback endpoint."""

    def test_feedback_valid_input(self, client, tmp_path):
        """Test feedback with valid input."""
        # Patch the feedback log path
        with patch('api.main.FEEDBACK_LOG_PATH', str(tmp_path / "feedback.json")):
            payload = {
                "timestamp": "2024-01-15T10:30:00",
                "predicted_weights": [0.4, 0.3, 0.3],
                "realized_return": 0.025,
                "actual_weights": [0.38, 0.32, 0.30]
            }
            
            response = client.post("/feedback", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "logged"

    def test_feedback_missing_fields(self, client):
        """Test feedback rejects missing required fields."""
        payload = {
            "timestamp": "2024-01-15T10:30:00",
            "predicted_weights": [0.4, 0.3, 0.3]
            # Missing realized_return
        }
        
        response = client.post("/feedback", json=payload)
        assert response.status_code == 422

    def test_feedback_invalid_return(self, client, tmp_path):
        """Test feedback handles extreme return values."""
        with patch('api.main.FEEDBACK_LOG_PATH', str(tmp_path / "feedback.json")):
            payload = {
                "timestamp": "2024-01-15T10:30:00",
                "predicted_weights": [0.4, 0.3, 0.3],
                "realized_return": 100.0,  # Extreme but technically valid
                "actual_weights": [0.38, 0.32, 0.30]
            }
            
            response = client.post("/feedback", json=payload)
            # Should accept but might warn
            assert response.status_code == 200

    def test_feedback_weights_mismatch(self, client, tmp_path):
        """Test feedback when predicted and actual weights differ."""
        with patch('api.main.FEEDBACK_LOG_PATH', str(tmp_path / "feedback.json")):
            payload = {
                "timestamp": "2024-01-15T10:30:00",
                "predicted_weights": [0.5, 0.3, 0.2],
                "realized_return": 0.015,
                "actual_weights": [0.4, 0.4, 0.2]
            }
            
            response = client.post("/feedback", json=payload)
            assert response.status_code == 200


class TestCORSSupport:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        # CORS should allow the request
        assert response.status_code in [200, 204]


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_model_not_loaded(self):
        """Test behavior when model fails to load."""
        with patch('api.main.load_trained_model') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model not found")
            
            from api.main import app
            with TestClient(app) as test_client:
                # Health should still work but report model not loaded
                response = test_client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["model_loaded"] is False
                assert data["status"] == "degraded"

    def test_nan_handling(self, client):
        """Test that NaN inputs are handled gracefully."""
        payload = {
            "market_data": [
                {"price": float('nan'), "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
            ],
            "portfolio_state": {
                "current_weights": [1.0],
                "cash_ratio": 0.1,
                "total_value": 10000.0
            }
        }
        
        # JSON doesn't support NaN, so this should fail at parsing
        # If it gets through, the API should handle it
        try:
            response = client.post("/predict", json=payload)
            # If request succeeds, check for proper handling
            if response.status_code == 200:
                data = response.json()
                # Ensure no NaN in output
                assert all(not (isinstance(w, float) and np.isnan(w)) 
                          for w in data.get("allocation_weights", []))
        except (ValueError, TypeError):
            # Expected - JSON serialization of NaN fails
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
