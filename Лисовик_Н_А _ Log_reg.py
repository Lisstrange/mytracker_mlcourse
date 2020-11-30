class LogReg:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.threshold = 0.5
        self.verbose = verbose
        self.losses = []
        self.w = []

    def add_intercept(self, X):
        print(len(X))
        X = np.append(X, [[1] for i in range(len(X))], axis=1)
        pass
        return X

    def sigmoid(self, z):
        pass
        return [1 / (1 + math.exp(-i)) for i in z]

    def log_loss(self, h, y):
        pass
        return -1 * sum([y[i] * math.log(h[i]) + (1 - y[i]) * math.log(1 - h[i]) for i in range(len(h))])

    def fit(self, X, y):
        w = np.zeros(X.shape[1] + 1)
        X = self.add_intercept(X)
        X_y = np.dot(X.T, (self.threshold - y))
        grad = 1 / len(y) * np.dot(X.T, (self.threshold - y))
        w = w - self.lr * grad
        losses = grad
        self.w = w
        return w, losses

    def predict_proba(self, X):
        pass
        X = self.add_intercept(X)
        return np.dot(X, self.w)

    @staticmethod
    def apply_sigm(X, h):
        array_pred = []
        for i in range(len(X)):
            if X[i] > h:
                array_pred.append(1)
            else:
                array_pred.append(0)
        return array_pred

    def predict(self, X):
        # print(X.shape)
        # print(self.w.shape)
        X = self.add_intercept(X)
        predict_proba = np.dot(X, self.w)
        # print(predict_proba)
        predict = self.sigmoid(predict_proba)
        # print(predict)
        predict = LogReg.apply_sigm(predict, self.threshold)
        pass
        return np.array(predict)

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        score_log = (y_test == preds).mean()
        pass
        return score_log



