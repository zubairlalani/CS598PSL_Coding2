import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import math

def bic_score(estimator, X, y):
    est = estimator.fit(X, y)
    n, p = X.shape
    y_pred = est.predict(X)
    rss = np.sum((y - y_pred) ** 2)
    bic = n * np.log(rss / n) + (p + 1) * np.log(n)
    return -bic

def main():
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

    train_size, test_size = 200, 800
    assert((train_size + test_size) == n)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size, random_state=42)
    assert(X_train.shape == (train_size,p))
    assert(X_test.shape == (test_size, p))
    assert(y_train.shape == (train_size,))
    assert(y_test.shape == (test_size,))

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
    


if __name__ == "__main__":
    main()