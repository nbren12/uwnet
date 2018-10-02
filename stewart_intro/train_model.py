from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def evaluate_linar_model(x_train, x_test, y_train, y_test):
    model = LinearRegression().fit(x_train, y_train)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    evaluate_linar_model(x_train, x_test, y_train, y_test)
