import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class linearRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    '''function for model training'''

    def fit(self, x, y):
        # m = training examples, n = no. of features
        self.m, self.n = x.shape
        # weight initialization
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

        '''gradient descent'''
        for i in range(self.iterations):
            self.update_weights()
        return self

    '''update weights'''

    def update_weights(self):
        y_pred = self.predict(self.x)
        # calculate gradients
        dw = -(2 * (self.x.T).dot(self.y - y_pred)) / self.m
        db = -2 * sum(self.y - y_pred) / self.m

        # update weights
        self.w = self.w - (self.learning_rate * dw)
        self.b = self.b - (self.learning_rate * db)
        return self

    '''predict'''

    def predict(self, x):
        return x.dot(self.w) + self.b

    # ''' ************ MAIN ************'''


def main():
    df = pd.read_csv('salary_data.csv')
    # df = dataset.copy()
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # model training
    model = linearRegression(iterations=1000, learning_rate=0.01)
    model.fit(x_train, y_train)

    # Prediction
    y_pred = model.predict(x_test)
    print(y_pred)

    print("Predicted values ", np.round(y_pred[:3], 2))
    print("Real values      ", y_test[:3])
    print("Trained W        ", round(model.w[0], 2))
    print("Trained b        ", round(model.b, 2))

    plt.scatter(x_train, y_train, color = 'blue')
    plt.plot(x_test, y_pred, color = 'red')
    plt.title('Salary vs Experience')
    plt.xlabel('Experience (Years)')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    main()
