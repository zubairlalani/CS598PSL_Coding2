import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from abess import LinearRegression as ALR

train_df = pd.read_csv('mls_train.csv')
train_x, train_y = train_df.drop('salary', axis=1), train_df['salary']

test_df = pd.read_csv('mls_test.csv')
test_x, test_y = test_df.drop('salary', axis=1), test_df['salary']

def mse(y, yhat): 
    d = np.asarray(yhat) - np.asarray(y)
    return float(np.mean(d * d))

def fit_lr(Xtr, ytr, Xte, yte):
    n = Xtr.shape[0]
    p = Xtr.shape[1] + 1 

    sklr = LinearRegression().fit(Xtr, ytr)
    yhat_tr = sklr.predict(Xtr)
    yhat_te = sklr.predict(Xte)

    RSS = float(np.sum((ytr - yhat_tr) ** 2))
    TSS = float(np.sum((ytr - np.mean(ytr)) ** 2))

    AIC = n * np.log(RSS / n) + 2 * p
    BIC = n * np.log(RSS / n) + np.log(n) * p
    R2_adj = 1.0 - (RSS / (n - p)) / (TSS / (n - 1))

    return {
        "ols": sklr,
        "RSS": RSS,
        "p": p,
        "train_mse": mse(ytr, yhat_tr),
        "test_mse": mse(yte, yhat_te),
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

# b)
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
print(results[["k","train_mse","test_mse","AIC","BIC","Cp","R2_adj","features"]])

best_AIC = results.loc[results["AIC"].idxmin()]
best_BIC = results.loc[results["BIC"].idxmin()]
best_Cp  = results.loc[results["Cp"].idxmin()]
best_R2a = results.loc[results["R2_adj"].idxmax()]

print("Best by AIC:\n", best_AIC[["k","features","train_mse","test_mse","AIC"]])
print("Best by BIC:\n", best_BIC[["k","features","train_mse","test_mse","BIC"]])
print("Best by Mallows Cp:\n", best_Cp[["k","features","train_mse","test_mse","Cp"]])
print("Best by adjusted R^2:\n", best_R2a[["k","features","train_mse","test_mse","R2_adj"]])

# c)




