from keras.optimizers import SGD, Adagrad


def compile_optimizer(name, model):
    optimizer = None
    if name == 'svg':
        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    elif name == 'adagrad':
        optimizer = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
