import numpy as np
import pytest
from ps3.evaluation import _evaluate_predictions as ep


@pytest.mark.parametrize(
    "predicted, actual, weights, expected",
    [
        # Case 1: All zeros
        (
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            {
                "bias": 0,
                "mae": 0,
                "rmse": 0
            },
        ),
        # Case 2: Predicted vs. actual with equal weights
        (
            np.array([1, 2]),
            np.array([1, 1]),
            np.array([1, 1]),
            {
                "bias": pytest.approx(0.5, rel=1e-2),  # Avg(Predicted - Actual) = (1 + 0) / 2
                "mae": pytest.approx(0.5, rel=1e-2),  # Mean(|Predicted - Actual|) = (0 + 1) / 2
                "rmse": pytest.approx(0.7071, rel=1e-2),  # Sqrt(Mean((Predicted - Actual)^2))
            },
        ),
        # Case 3: More complex case with weights
        (
            np.array([1, 2, 3]),
            np.array([1, 3, 0]),
            np.array([1, 2, 0]),
            {
                "bias": pytest.approx(-0.666, rel=1e-2),  # Weighted Avg(Predicted - Actual)
                "mae": pytest.approx(0.6667, rel=1e-2),  # Weighted Mean(|Predicted - Actual|)
                "rmse": pytest.approx(0.8165, rel=1e-2),  # Weighted RMSE
            },
        ),
    ],
)
def test_evaluate_predictions(predicted, actual, weights, expected):
    # Call the evaluate_predictions function
    result = ep.evaluate_predictions(predicted, actual, weights)
    
    # Assert results for each metric
    assert result["bias"] == expected["bias"]
    assert result["mae"] == expected["mae"]
    assert result["rmse"] == expected["rmse"]
