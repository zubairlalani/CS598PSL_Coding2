import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import KFold
from abess import LinearRegression as ALR

train_df = pd.read_csv('mls_train.csv')
train_x, train_y = train_df.drop('salary', axis=1), train_df['salary']

test_df = pd.read_csv('mls_test.csv')
test_x, test_y = test_df.drop('salary', axis=1), test_df['salary']

def mse(y, yhat): 
    d = np.asarray(yhat) - np.asarray(y)
    return float(np.mean(d * d))

def fit_lr(train_x, train_y, test_x, test_y):
    n = train_x.shape[0]
    p = train_x.shape[1] + 1 

    sklr = LinearRegression().fit(train_x, train_y)
    pred_train_y = sklr.predict(train_x)
    pred_test_y = sklr.predict(test_x)

    RSS = float(np.sum((train_y - pred_train_y) ** 2))
    TSS = float(np.sum((train_y - np.mean(train_y)) ** 2))

    AIC = n * np.log(RSS / n) + 2 * p
    BIC = n * np.log(RSS / n) + np.log(n) * p
    R2_adj = 1.0 - (RSS / (n - p)) / (TSS / (n - 1))

    return {
        "ols": sklr,
        "RSS": RSS,
        "p": p,
        "train_mse": mse(train_y, pred_train_y),
        "test_mse": mse(test_y, pred_test_y),
        "AIC": AIC,
        "BIC": BIC,
        "R2_adj": R2_adj,
    }

# a)
best_subsets = {} 
feature_names = train_x.columns.to_numpy()

for k in range(1, 9):
    m = ALR(support_size=k) 
    m.fit(train_x, train_y)
    idx = np.nonzero(m.coef_)[0]
    best_subsets[k] = feature_names[idx].tolist()

lr_dict = fit_lr(train_x[best_subsets[max(best_subsets)]], train_y, test_x[best_subsets[max(best_subsets)]], test_y)
sigma2_hat = lr_dict["RSS"] / (train_x.shape[0] - lr_dict["p"]) 
rows = []

for k, feats in best_subsets.items():
    lr_dict = fit_lr(train_x[feats], train_y, test_x[feats], test_y)
    Cp = lr_dict["RSS"] / sigma2_hat - (train_x.shape[0] - 2 * lr_dict["p"]) 
    rows.append({
        "k": k,
        "features": feats,
        "train_mse": lr_dict["train_mse"],
        "test_mse": lr_dict["test_mse"],
        "AIC": lr_dict["AIC"],
        "BIC": lr_dict["BIC"],
        "Cp": Cp,
        "R2_adj": lr_dict["R2_adj"],
    })

results = pd.DataFrame(rows).sort_values("k")
print(results[["k","train_mse","test_mse", "features"]])

# b)
best_AIC = results.loc[results["AIC"].idxmin()]
best_BIC = results.loc[results["BIC"].idxmin()]
best_Cp  = results.loc[results["Cp"].idxmin()]
best_R2a = results.loc[results["R2_adj"].idxmax()]

print("Best by AIC:\n", best_AIC[["k","features","train_mse","test_mse","AIC"]])
print("Best by BIC:\n", best_BIC[["k","features","train_mse","test_mse","BIC"]])
print("Best by Mallows Cp:\n", best_Cp[["k","features","train_mse","test_mse","Cp"]])
print("Best by adjusted R^2:\n", best_R2a[["k","features","train_mse","test_mse","R2_adj"]])

# c)
print("Ridge: ")
cv = KFold(n_splits = 5, shuffle = True, random_state=42) # 5-fold cross validation
ridge = RidgeCV(cv = cv, scoring='neg_mean_squared_error')
ridge.fit(train_x, train_y)

print("Optimal lambda (alpha):", ridge.alpha_)
pred_train_y = ridge.predict(train_x)
pred_test_y = ridge.predict(test_x)
print("Train MSE:", mse(train_y, pred_train_y))
print("Test  MSE:", mse(test_y, pred_test_y))

# d)
print("Lasso: ")
lasso_cv = LassoCV(cv=cv)
lasso_cv.fit(train_x, train_y)
lasso = Lasso(alpha = lasso_cv.alpha_)
lasso.fit(train_x, train_y)
coef_series = pd.Series(lasso.coef_, index=train_x.columns)
zero_feature_indices = np.isclose(coef_series.values, 0.0)
zero_features = coef_series.index[zero_feature_indices].tolist()
print(zero_features)
train_pred_y = lasso.predict(train_x)
test_pred_y = lasso.predict(test_x)
print("Train MSE:", mse(train_y, train_pred_y))
print("Test MSE:", mse(test_y, test_pred_y))

# e)
'''
Based on the results, we can see that the mean squared error for both train and test is lowest when using ridge regression.
'''



