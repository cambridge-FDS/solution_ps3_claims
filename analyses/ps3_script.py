# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]

df = create_sample_split(df, id_column="IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# Let's add splines for BonusMalus and Density and use a pipeline

# Let's put together pipeline
numeric_cols = ["BonusMalus", "Density"]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("spline", SplineTransformer(include_bias=False, knots="quantile")),
                ]
            ),
            numeric_cols,
        ),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=1, fit_intercept=True
            ),
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# Let's use a GBM instead as an estimator
model_pipeline = Pipeline([("estimate", LGBMRegressor(objective="tweedie"))])

model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# Let's tune the pipeline to reduce overfitting

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned
cv = GridSearchCV(
    model_pipeline,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        "estimate__n_estimators": [50, 100, 150, 200],
    },
    verbose=2,
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


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


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
#Ex1
# Step 1: Create a plot of the average claims per BonusMalus group
weighted_avg_claims = (
    df.groupby("BonusMalus")
       .apply(lambda group: np.average(group["PurePremium"], weights=group["Exposure"])))

plt.figure(figsize=(10, 6))
plt.plot(weighted_avg_claims.index, weighted_avg_claims, marker=".")
plt.title("Weighted Average Claims per BonusMalus Group")
plt.xlabel("BonusMalus")
plt.ylabel("Weighted Average Claims")
plt.grid()
plt.show()

# %%
# Create a new model pipeline with monotonic constraints
# Define numeric columns and predictors again (make sure it fits here, although this has been done before)
numeric_cols = ["BonusMalus", "Density"]
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
predictors = categoricals + numeric_cols
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

# Fit the preprocessor to the training data
transformed_data = preprocessor.fit_transform(df_train[predictors])
num_features = transformed_data.shape[1]

# Define monotone constraints
monotone_constraints = [1 if "BonusMalus" in feature else 0 for feature in preprocessor.get_feature_names_out()]

# Define the constrained LGBM pipeline
constrained_lgbm = Pipeline([
    ("preprocess", preprocessor),
    ("estimate", LGBMRegressor(objective="tweedie", monotone_constraints=monotone_constraints))
])

# Perform GridSearchCV with the constrained LGBM
param_grid = {
    "estimate__learning_rate": [0.01, 0.02],
    "estimate__n_estimators": [100, 150],
}
cv = GridSearchCV(constrained_lgbm, param_grid, verbose=2)
cv.fit(df_train[predictors], y.iloc[train], estimate__sample_weight=df_train["Exposure"].values)


#%%
# Cross-validate and predict using the best estimator
df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(df_test[predictors])
df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(df_train[predictors])

def tweedie_deviance(y_true, y_pred, sample_weight):
    return np.sum(sample_weight * (y_true ** (2 - 1.5) / (2 - 1.5) - y_true * y_pred ** (1 - 1.5) / (1 - 1.5) + y_pred ** (2 - 1.5) / (2 - 1.5)))

print(
    "training loss t_lgbm_constrained:  {}".format(
        tweedie_deviance(y.iloc[train], df_train["pp_t_lgbm_constrained"], df_train["Exposure"]) / np.sum(df_train["Exposure"])
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        tweedie_deviance(y.iloc[test], df_test["pp_t_lgbm_constrained"], df_test["Exposure"]) / np.sum(df_test["Exposure"])
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %% 
# Ex2 Plot the learning curve
import lightgbm as lgb
best_lgbm_estimator = cv.best_estimator_.named_steps["estimate"]


best_lgbm_estimator.fit(
    transformed_data,  # The transformed training data
    y.iloc[train],
    sample_weight=df_train["Exposure"].values,
    eval_set=[(transformed_data, y.iloc[train]), (preprocessor.transform(df_test[predictors]), y.iloc[test])],
    eval_metric="tweedie",
    eval_names=["train", "validation"],
)

# Plot the learning curve for Tweedie deviance
lgb.plot_metric(best_lgbm_estimator.evals_result_, metric='tweedie')
plt.xlabel('Boosting Round')
plt.ylabel('tweedie')
plt.title('Learning Curve (tweedie Deviance)')
plt.grid()
plt.show()


# %%
