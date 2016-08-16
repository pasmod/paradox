from sklearn.cross_validation import train_test_split


def split_training_data(X, ygi):
    random_state = 123456
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    return {'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test_': y_test}
