import numpy as np
from glum import TweedieDistribution

def bias(predicted_values: np.array, actual_values: np.array, weights: np.array) -> float:
    """
    Calculate the bias of a model.

    Args:
    predicted_values: np.array
        The predicted values of the model.
    actual_values: np.array
        The actual values of the target variable.
    weights: np.array
        The weights of the observations.

    Returns:
    float
        The bias of the model.
    """
    bias = np.sum(weights * (predicted_values - actual_values)) / np.sum(weights)
    return float(bias)

def mean_absolute_error(predicted_values: np.array, actual_values: np.array, weights: np.array) -> float:
    """
    Calculate the mean absolute error of a model.

    Args:
    predicted_values: np.array
        The predicted values of the model.
    actual_values: np.array
        The actual values of the target variable.
    weights: np.array
        The weights of the observations.

    Returns:
    float
        The mean absolute error of the model.
    """
    mean_absolute_error = np.sum(weights * np.abs(predicted_values - actual_values)) / np.sum(weights)
    return float(mean_absolute_error)

def root_mean_squared_error(predicted_values: np.array, actual_values: np.array, weights: np.array) -> float:
    """
    Calculate the root mean squared error of a model.

    Args:
    predicted_values: np.array
        The predicted values of the model.
    actual_values: np.array
        The actual values of the target variable.
    weights: np.array
        The weights of the observations.

    Returns:
    float
        The root mean squared error of the model.
    """
    root_mean_squared_error = np.sqrt(np.sum(weights * (predicted_values - actual_values)**2) / np.sum(weights))
    return float(root_mean_squared_error)

def gini_coefficient(predicted_values: np.array, actual_values: np.array, weights: np.array) -> float:
    """
    Calculate the Gini coefficient of a model.

    Args:
    predicted_values: np.array
        The predicted values of the model.
    actual_values: np.array
        The actual values of the target variable.
    weights: np.array
        The weights of the observations.

    Returns:
    float
        The Gini coefficient of the model.
    """
    # Sort the values by the predicted values
    sorted_values = np.array(sorted(zip(predicted_values, actual_values, weights), key=lambda x: x[0], reverse=True))
    sorted_predicted_values = sorted_values[:, 0]
    sorted_actual_values = sorted_values[:, 1]
    sorted_weights = sorted_values[:, 2]

    # Divide the cumulative sum of the actual values by the sum of the actual values
    cumulative_actual_values = np.cumsum(sorted_weights * sorted_actual_values) / np.sum(sorted_weights * sorted_actual_values)
    cumulative_predicted_values = np.cumsum(sorted_weights * sorted_predicted_values) / np.sum(sorted_weights * sorted_predicted_values)

    # Get the straight line
    straight_line = np.linspace(0, 1, len(cumulative_actual_values))

    # Calculate the sum of (cumulative_actual_values - straight_line) area between the straight line and the actual values
    area_actual_vs_straight_line = sum(cumulative_actual_values - straight_line)
    area_predicted_vs_straight_line = sum(cumulative_predicted_values - straight_line)

    # Calculate the Gini coefficient
    gini_coefficient = area_predicted_vs_straight_line / area_actual_vs_straight_line

    return float(gini_coefficient)


def evaluate_predictions(
        predicted_values: np.array,
        actual_values: np.array,
        weights: np.array,
        tweedie_distribution: int = 1.5) -> dict:
    """
    Evaluate the predictions of a model.

    Args:
    predicted_values: np.array
        The predicted values of the model.
    actual_values: np.array
        The actual values of the target variable.
    weights: np.array
        The weights of the observations.
    """
    model_evaluation = {
        'bias': bias(predicted_values, actual_values, weights),
        'deviance': TweedieDistribution(tweedie_distribution).deviance(actual_values, predicted_values, weights),
        'mae': mean_absolute_error(predicted_values, actual_values, weights),
        'rmse': root_mean_squared_error(predicted_values, actual_values, weights),
        'gini': gini_coefficient(predicted_values, actual_values, weights)
    }
    return model_evaluation


