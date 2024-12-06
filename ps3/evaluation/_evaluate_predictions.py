import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc


def evaluate_predictions(
    df,
    outcome_column,
    *,
    preds_column=None,
    model=None,
    tweedie_power=1.5,
    exposure_column=None,
):
    """Evaluate predictions against actual outcomes.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe used for evaluation
    outcome_column : str
        Name of outcome column
    preds_column : str, optional
        Name of predictions column, by default None
    model :
        Fitted model, by default None
    tweedie_power : float, optional
        Power of tweedie distribution for deviance computation, by default 1.5
    exposure_column : str, optional
        Name of exposure column, by default None

    Returns
    -------
    evals
        DataFrame containing metrics
    """

    evals = {}

    assert (
        preds_column or model
    ), "Please either provide the column name of the pre-computed predictions or a model to predict from."

    if preds_column is None:
        preds = model.predict(df)
    else:
        preds = df[preds_column]

    if exposure_column:
        weights = df[exposure_column]
    else:
        weights = np.ones(len(df))

    evals["mean_preds"] = np.average(preds, weights=weights)
    evals["mean_outcome"] = np.average(df[outcome_column], weights=weights)
    evals["bias"] = (evals["mean_preds"] - evals["mean_outcome"]) / evals[
        "mean_outcome"
    ]

    evals["mse"] = np.average((preds - df[outcome_column]) ** 2, weights=weights)
    evals["rmse"] = np.sqrt(evals["mse"])
    evals["mae"] = np.average(np.abs(preds - df[outcome_column]), weights=weights)
    evals["deviance"] = TweedieDistribution(tweedie_power).deviance(
        df[outcome_column], preds, sample_weight=weights
    ) / np.sum(weights)
    ordered_samples, cum_actuals = lorenz_curve(df[outcome_column], preds, weights)
    evals["gini"] = 1 - 2 * auc(ordered_samples, cum_actuals)

    return pd.DataFrame(evals, index=[0]).T


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount
