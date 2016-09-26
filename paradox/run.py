from keras.callbacks import ModelCheckpoint
from encoders import models
from corpus import loader
import numpy as np
np.set_printoptions(threshold=np.inf)


(X_train_1, X_train_2, y_train), (X_test_1, X_test_2, y_test), vocab, embeddings, weights = loader.load(n=21000, lang='en')  # noqa

max_len = len(X_train_1[0])


def run_lstm_encoder(nb_epoch=100, batch_size=64, encoding_dim=64, depth=3,
                     activation='relu', embeddings=None, weights=None,
                     lang='en'):
    model = models.lstm_encoder(encoding_dim=encoding_dim,
                                depth=depth, activation=activation,
                                max_len=max_len, vocab=vocab,
                                embeddings=embeddings, weights=weights)
    checkpoint = ModelCheckpoint("lstm_encoder.model."
                                 "{epoch:03d}-{val_acc:.4f}.hdf5",
                                 monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    model.fit([X_train_1, X_train_2], y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, verbose=1, shuffle=True,
              validation_data=([X_test_1, X_test_2], y_test),
              callbacks=[checkpoint])
    y_pred = model.predict([X_test_1, X_test_2])
    evaluate_keras_predictions(y_test, y_pred)


def calculate_and_print_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from pandas_confusion import ConfusionMatrix
    accuracy = accuracy_score(y_true, y_pred)
    print("*** Results ***" + '*' * 5)
    print("Accuracy:" + str(accuracy))
    f1 = f1_score(y_true, y_pred, average=None)
    print("Macro F1:" + str(f1))
    avg_f1 = str(np.mean(f1))
    print("Average F1:" + avg_f1)
    print("Confusion matrix")
    print(ConfusionMatrix(y_true, y_pred))
    return avg_f1


def evaluate_keras_predictions(y_true, y_pred):
    y_pred_new = []
    y_true_new = []
    if len(y_pred[0]) == 2:
        for y in y_pred:
            if y[0] > y[1]:
                y_pred_new.append('1.0, 0.0')
            else:
                y_pred_new.append('0.0, 1.0')
    if len(y_pred[0]) == 3:
        for y in y_pred:
            if y[0] > y[1] and y[0] > y[2]:
                y_pred_new.append('1.0, 0.0, 0.0')
            if y[1] > y[0] and y[1] > y[2]:
                y_pred_new.append('0.0, 1.0, 0.0')
            if y[2] > y[1] and y[2] > y[0]:
                y_pred_new.append('0.0, 0.0, 1.0')
    if len(y_true[0]) == 2:
        for y in y_true:
            if y[0] > y[1]:
                y_true_new.append('1.0, 0.0')
            else:
                y_true_new.append('0.0, 1.0')
    if len(y_true[0]) == 3:
        for y in y_true:
            if y[0] > y[1] and y[0] > y[2]:
                y_true_new.append('1.0, 0.0, 0.0')
            if y[1] > y[0] and y[1] > y[2]:
                y_true_new.append('0.0, 1.0, 0.0')
            if y[2] > y[1] and y[2] > y[0]:
                y_true_new.append('0.0, 0.0, 1.0')
    return calculate_and_print_metrics(y_true_new, y_pred_new)


# run_simple_encoder(activation='softmax', batch_size=64, nb_epoch=20)
# run_deep_encoder(depth=10)
for i in range(0, 10):
    print("Iteration {}".format(i))
    run_lstm_encoder(nb_epoch=15, batch_size=64, embeddings=embeddings,
                     lang='en', weights=weights)
