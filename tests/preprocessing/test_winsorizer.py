import numpy as np
import pytest

from ps3.preprocessing import Winsorizer


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_square_transformer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    Xt = Winsorizer(
        lower_quantile=lower_quantile, upper_quantile=upper_quantile
    ).fit_transform(X)

    assert (Xt.max() == np.quantile(X, q=upper_quantile)) & (
        Xt.min() == np.quantile(X, q=lower_quantile)
    )
