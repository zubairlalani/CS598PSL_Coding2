## Coding Assignment 2

### Project Members:
Zubair Lalani (netid: zubairl2)
Adithya Swaminathan (netid: adithya9)

## Problem 1

### 1a

We begin by import all necessary packages

```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import math
```

Below, we address **1a** by generating a data set with $p=20$ and $n=1000$ observations according the penalized linear model specified. In addition, we define our constants.

```
n = 1000
p = 20  
mu, sigma = 0, 1

rng = np.random.default_rng(seed=42)
X = rng.normal(mu, sigma, (n, p))
e = rng.normal(mu, sigma, n)
B = np.array([1,0.5,0,-0.5,-1,1,0.5,2,0,0,0.1,0.2, 2, 0, 0, 0, -2, 1, 0, 0])
Y = X @ B + e

assert(B.shape == (p,))
assert(X.shape == (n, p))
assert(e.shape == (n,))
assert(Y.shape == (n,))
```

### 1b

Now we can randomly split our data into training and test set using the constants specified int he problem statement.

```
train_size, test_size = 200, 800
assert((train_size + test_size) == n)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size, random_state=42)
assert(X_train.shape == (train_size,p))
assert(X_test.shape == (test_size, p))
assert(y_train.shape == (train_size,))
assert(y_test.shape == (test_size,))
```

### Code for 1c-1g

Parts 1c-1g are tightly coupled within the code so we will provide the code for all of these portions together, but provide any further required explanation and resulting plots using the following sections.

```
def bic_score(estimator, X, y):
	est = estimator.fit(X, y)
	n, p = X.shape
	y_pred = est.predict(X)
	rss = np.sum((y - y_pred) ** 2)
	bic = n * np.log(rss / n) + (p + 1) * np.log(n)
	return -bic
```

```
lr = LinearRegression()
mse_train = [-1] * p
mse_test = [-1] * p
coeffs_err = [-1] * p
coeffs_err_test = [-1] * p
B_hats = np.zeros((n,p))
for k in range(1, p):
	sfs = SequentialFeatureSelector(lr, n_features_to_select=k,direction='forward', scoring=bic_score)
	sfs.fit(X_train, y_train)
	features_selected = sfs.get_support()
	assert(sum(features_selected) == k)

	X_subset = X_train[:, features_selected]
	lr.fit(X_subset, y_train)
	y_pred = lr.predict(X_subset)
	loss = mean_squared_error(y_train, y_pred)

	assert(loss >= 0)
	mse_train[k-1] = loss

	X_test_subset = X_test[:, features_selected]
	y_pred_test = lr.predict(X_test_subset)
	loss = mean_squared_error(y_test, y_pred_test)
	assert(loss >= 0)
	mse_test[k-1] = loss

	B_hat_r = np.zeros((p,))
	B_hat_r[features_selected] = lr.coef_
	
	coeffs_err[k-1] = np.sqrt(np.sum(np.square((B - B_hat_r))))

	B_hats[k-1] = B_hat_r.T
# Need to handle the last index (where all coefficients are included) separately
# because SequentialFeatureSelector seems to error out when specifiying n_features_to_select=p
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
loss = mean_squared_error(y_train, y_pred)
assert(loss >= 0)
mse_train[-1] = loss

y_pred_test = lr.predict(X_test)
loss = mean_squared_error(y_test, y_pred_test)

assert(loss >= 0)
mse_test[-1] = loss
coeffs_err[-1] = np.sum(np.square(B - lr.coef_))

print("Full model coeffs:")
print(lr.coef_)

print("Coefficients errors (1g): ")
print(coeffs_err)

B_hats[-1] = lr.coef_
p_min = np.argmin(mse_test) + 1 # num coefficients for which the MSE error on on test set is minimized
# print("MSE Error Test: ")
print(mse_test)
print("Num coefficients for which MSE error minimized on test set: ", p_min)
print("Corresponding Coefficients for predictive model: ")
print(B_hats[p_min-1])
print("True Coefficients:")
print(B)

sizes = range(1,21)
plt.figure(1)
plt.plot(sizes, mse_train, label="train", color="orange")
plt.plot(sizes, mse_test, label="test", color="blue")

plt.xlabel("Model Size")
plt.ylabel("MSE Error")
plt.title("MSE Error vs Model Size")
plt.xticks(sizes)
plt.legend(loc="upper right")

plt.figure(2)
plt.plot(sizes, coeffs_err)
plt.xlabel("Model Size")
plt.ylabel("Coefficients Error")
plt.title("Coefficient Error vs Model Size (1g)")
plt.xticks(sizes)
plt.show()
```

### 1c-1d Results

Our resulting plot for **1c** and **1d** is shown below. The MSE error on the training set for best model of each size is shown in orange, while the MSE error on the test set for the best model of each size is shown in blue.

![[Screenshot 2025-09-21 at 1.20.57 PM.png]]
### 1e
Based on the resulting plot from the previous section, a model size of **12** coefficients results in a minimum MSE on the test set. This result is interesting as we know that the true model uses **p=20** beta coefficients, but these results seem to suggest that we can achieve a good accuracy  with only **12** coefficients. We use the BIC score criterion during our subset selection procedure for each size. Since the true model actually has a value of 0 for 8 of the coefficients, it makes sense that a model size in the range of $10-20$ seems to result in similar results as the linear regression likely ends up predicting very small coefficients for the larger models $p > 12$ anyways.

![[Screenshot 2025-09-20 at 2.11.28 PM.png]]

### 1f

The model at which the test set MSE is minimized uses 12 coefficients. Interestingly, the true model also technically used 12 coefficients even though we had $p=20$  because 8 of the coefficients from the true model had a value of 0 and thus 8 of the covariates had no real impact on the data generated.

Below are the values of the $\beta$ coefficients from the true model:

```
β = (1, 0.5, 0, −0.5, −1, 1, 0.5, 2, 0, 0, 0.1, 0.2, 2, 
0, 0, 0, −2, 1, 0, 0)
```

See below for the value of the coefficients of the model that minimized the MSE on the test data. Note that we have the nonzero values for the coefficients that this model chose ($p=12$):

```
β = (1.089, 0.449, 0.140, -0.429, -1.031, 1.053, 0.631, 1.921, 0, 0, 0, 0.150, 1.986, 0,0,0,-2.007, 1.034,0,0)
```

The true model and our predicted model have quite similar coefficient values. The forward selection procedure seemed to use pretty much all of the variables. It seems the main difference is that the true model did not use the 3rd predictor in the data , $B_3=0$, and instead used the 11th predictor, $B_{11} \ne 0$. However, the actual coefficient values for these 2 were small in our model regardless so this difference didn't have too much of an impact on the predictions. In addition, each individual $B_i$ and $\hat{B_i}$ that both the true and predicted models used were rather similar.

### 1g

The actual error between the coefficient values follows a similar trajectory as the MSE plot from **(1d)**. We can see that in this plot, the best model of size $p=12$ seems to minimize the error between the predicted coefficients and the true model's coefficients. Technically, this plot shows that the best model of size $p=20$ had the least error between the coefficients; however, this is expected since we are allowing the linear regression model to have access to all of the data. But, all the models of size $p>10$ in this plot result in the coefficients that are pretty close to the true coefficients.

![[Screenshot 2025-09-21 at 1.20.42 PM.png]]


## Problem 2

### 2a

Below is our code for performing the best subset selection, computing the prediction errors on the training/testing sets, as well as printing the results of the "best model" for each size.

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from abess import LinearRegression as ALR

train_df = pd.read_csv('mls_train.csv')
train_x, train_y = train_df.drop('salary', axis=1), train_df['salary']

test_df = pd.read_csv('mls_test.csv')
test_x, test_y = test_df.drop('salary', axis=1), test_df['salary']
```

```
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
        "train_mse": mse(train_y, pred_train_y),
        "test_mse": mse(test_y, pred_test_y),
        "AIC": AIC,
        "BIC": BIC,
        "RSS": RSS,
        "p": p,
        "R2_adj": R2_adj,
    }
```

```
best_subsets = {} 
feature_names = train_x.columns.to_numpy()

for k in range(1, 9):
    m = ALR(support_size=k) 
    m.fit(train_x, train_y)
    idx = np.nonzero(m.coef_)[0]
    best_subsets[k] = feature_names[idx].tolist()
    print(f"Best model of size {k} uses the following features:")
    print(best_subsets[k])

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
pd.set_option("display.max_colwidth", None)
print(results[["k","train_mse","test_mse", "features"]])
```

Below we report the results of the predictions errors and the features that were selected for the best model of size $k$.

```
k     train_mse      test_mse
1  3.258455e+11  6.521076e+11
2  2.989997e+11  6.748624e+11
3  2.839730e+11  6.637367e+11
4  2.765923e+11  6.613063e+11
5  2.692579e+11  6.579558e+11
6  2.625697e+11  6.802184e+11
7  2.590164e+11  6.792259e+11
8  2.593450e+11  6.731992e+11
```

```
k  features
1  [ontarget_scoring_att]
2  [ontarget_scoring_att, successful_short_pass]
3  [goals, aerial_won, successful_short_pass]
4  [goals, won_tackle, aerial_won, successful_short_pass]
5  [goals, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass]
6  [weight, kg, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
7  [weight, kg, duel_won, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
8  [weight, kg, sub_on, accurate_cross, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
```
### 2b

Below is the code for calculating the AIC, BIC, and $C_p$-Mallows, and $R_{a}^2$ values. For each criterion, we have included the MSE on the training/testing data.

```
best_AIC = results.loc[results["AIC"].idxmin()]
best_BIC = results.loc[results["BIC"].idxmin()]
best_Cp  = results.loc[results["Cp"].idxmin()]
best_R2a = results.loc[results["R2_adj"].idxmax()]

print("Best by AIC:\n", best_AIC[["k","features","train_mse","test_mse","AIC"]])
print("Best by BIC:\n", best_BIC[["k","features","train_mse","test_mse","BIC"]])
print("Best by Mallows Cp:\n", best_Cp[["k","features","train_mse","test_mse","Cp"]])
print("Best by adjusted R^2:\n", best_R2a[["k","features","train_mse","test_mse","R2_adj"]])
```

See the results below. Each snippet lists the criterion used (AIC, BIC, Mallows CP, or adjusted $R^2$$, the resulting model size $k$, features selected, Mean Squared Error (MSE) on the training data, Mean Squared Error (MSE) on the testing data, and the criterion score.
``
```
Best by AIC:
k       7
features     [weight, kg, duel_won, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
train_mse     259016392626.879913
test_mse      679225904010.675659
AIC           9371.735959
```

```
Best by BIC:
k      6
features     [weight, kg, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
train_mse   262569679499.918274
test_mse    680218400338.292358
BIC         9401.711024
```

```
Best by Mallows Cp:
k     7
features     [weight, kg, duel_won, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
train_mse    259016392626.879913
test_mse     679225904010.675659
Cp           6.560337
```

```
Best by adjusted R^2:
k      7
features     [weight, kg, duel_won, won_tackle, aerial_won, ontarget_scoring_att, successful_short_pass, total_offside]
train_mse    259016392626.879913
test_mse     679225904010.675659
R2_adj       0.432486
```

### 2c

In part c, we fit a **ridge** regression model to predict a player's salary. Then, use cross-validation to select the best regularization parameter $\lambda$

The code below shows our approach:

```
print("Ridge: ")
scaler = StandardScaler().fit(train_x)
alphas = np.logspace(-4, 4, 100)  
scaled_tr_x = scaler.transform(train_x)
scaled_test_x = scaler.transform(test_x)
cv = KFold(n_splits = 5, shuffle = True, random_state=42) # 5-fold cross validation
ridge = RidgeCV(alphas = alphas, cv = cv, scoring='neg_mean_squared_error')
ridge.fit(scaled_tr_x, train_y)

print("Optimal lambda (alpha):", ridge.alpha_)
pred_train_y = ridge.predict(scaled_tr_x)
pred_test_y = ridge.predict(scaled_test_x)
print("Train MSE:", mse(train_y, pred_train_y))
print("Test  MSE:", mse(test_y, pred_test_y))
```

See below for the results:

```
Ridge: 
Optimal lambda (alpha): 25.950242113997373
Train MSE: 253016991144.07742
Test  MSE: 638314094075.6615
```

### 2d
In part d, we fit a lasso regression model on the same data set and identify the features that shrunk to zero.

Our code is provided below:

```
print("Lasso: ")
scaler = StandardScaler().fit(train_x)
scaled_tr_x = scaler.transform(train_x)
scaled_test_x = scaler.transform(test_x)
lasso_cv = LassoCV(alphas = alphas, cv=cv)
lasso_cv.fit(scaled_tr_x, train_y)
lasso = Lasso(alpha = lasso_cv.alpha_)
lasso.fit(scaled_tr_x, train_y)
coef_series = pd.Series(lasso.coef_, index=train_x.columns)
zero_feature_indices = np.isclose(coef_series.values, 0.0)
zero_features = coef_series.index[zero_feature_indices].tolist()
print(zero_features)
train_pred_y = lasso.predict(scaled_tr_x)
test_pred_y = lasso.predict(scaled_test_x)
print("Train MSE:", mse(train_y, train_pred_y))
print("Test MSE:", mse(test_y, test_pred_y))
```

See below for our results. The features listed in the second line are the features that shrunk to zero.

```
Lasso: 
['height, cm', 'game_started', 'mins', 'yellow_card']
Train MSE: 254427267696.63605
Test MSE: 650215622813.5703
```

### 2e
Based on the results, we can see that the mean squared error for both train and test is lowest when using ridge regression. This is because 
most of the features are important when determining the salary a player gets in relation to their statistics. Ridge regression, as opposed to lasso regression, 
assumes that most features are important when it goes into factors that influence a player's salary (minutes, goals, assists, games_started, height, etc.). As
a result, it makes sense that ridge regression has a lower mean squared error on average as opposed to a feature-selection based regression such as 
lasso. Furthermore, since this is the case, it makes sense that measures such as AIC and BIC (which penalize complexity of features) also yield high
mean squared errors.