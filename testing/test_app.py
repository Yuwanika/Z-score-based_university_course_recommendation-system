import unittest
from unittest.mock import patch, MagicMock
from app import app
import json
import numpy as np

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        # Creates a test client
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.stage1_model')
    @patch('app.stage2_model')
    @patch('app.label_encoders')
    @patch('app.scaler')
    @patch('app.poly')
    def test_predict_endpoint(self, mock_poly, mock_scaler, mock_label_encoders, mock_stage2_model, mock_stage1_model):
        # Mock label encoders
        mock_label_encoders.__getitem__.side_effect = lambda key: {
            'stream': MagicMock(transform=lambda x: [1]),
            'district': MagicMock(transform=lambda x: [2]),
            'stream_district': MagicMock(transform=lambda x: [3]),
            'university_course': MagicMock(
                inverse_transform=lambda x: np.array([f"Uni{x_i}|Course{x_i}" for x_i in x])
            )
        }[key]

        # Mock polynomial features
        mock_poly.transform.return_value = [[0.9, 0.81]]

        # Mock scaler
        mock_scaler.transform.return_value = [[0.9, 1.8, 0.81, 0.0]]

        # Mock stage1 model
        mock_stage1_model.predict_proba.return_value = [np.linspace(0, 1, 100)]

        # Mock stage2 model
        mock_stage2_model.predict_proba.return_value = np.random.rand(10, 2)

        # Sample input
        payload = {
            'z_score': 1.98,
            'stream': 'BIOLOGICAL SCIENCE',
            'district': 'KANDY',
            'year': 2025
        }

        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Ensure keys exist
        self.assertIn('university', data)
        self.assertIn('course', data)

        print("Tested /predict endpoint with mock models. Response:")
        print(data)

if __name__ == '__main__':
    unittest.main()
