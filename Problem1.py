import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt

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
    train_size, test_size = 200, 800
    assert((train_size + test_size) == n)

    rng = np.random.default_rng()
    X = rng.normal(mu, sigma, (n, p))
    e = rng.normal(mu, sigma, n)
    B = np.array([1,0.5,0,-0.5,-1,1,0.5,2,0,0,0.1,0.2, 2, 0, 0, 0, -2, 1, 0, 0])
    Y = X @ B + e

    assert(B.shape == (p,))
    assert(X.shape == (n, p))
    assert(e.shape == (n,))
    assert(Y.shape == (n,))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)
    assert(X_train.shape == (train_size,p))
    assert(X_test.shape == (test_size, p))
    assert(y_train.shape == (train_size,))
    assert(y_test.shape == (test_size,))

    lr = LinearRegression()
    mse_train = [-1] * p
    mse_test = [-1] * p
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

        
        print(lr.coef_)

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    loss = mean_squared_error(y_train, y_pred)
    assert(loss >= 0)
    mse_train[-1] = loss

    y_pred_test = lr.predict(X_test)
    loss = mean_squared_error(y_test, y_pred_test)
    
    assert(loss >= 0)
    mse_test[-1] = loss

    sizes = range(1,21)
    plt.plot(sizes, mse_train, label="train", color="orange")
    plt.plot(sizes, mse_test, label="test", color="blue")

    plt.xlabel("Model Size")
    plt.ylabel("MSE Error")
    plt.title("MSE Error vs Model Size")

    plt.legend(loc="upper right")

    plt.show()

    


if __name__ == "__main__":
    main()