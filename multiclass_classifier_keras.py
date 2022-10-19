import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_data(k: int, npoints: int):
    """
    Create random data
     k: number of clusters/classes
     npoints: total number of points to be generated
    """
    np.random.seed(1)
    X = []
    labels = []
    for i in range(k):
        center_year = np.random.uniform(0, 100)
        center_income = np.random.uniform(0, 1000000)
        npoints = int(npoints / k)
        for j in range(npoints):
            labels.append(i)
            X.append([np.random.normal(center_year, 1), np.random.normal(center_income, 10000)])
    return np.array(X), np.array(labels)


def make_scatter_plot(data, labels, k, outname):
    """ Make scatter plot"""
    colors = {
        0: 'red',
        1: 'black',
        2: 'blue',
        3: 'green',
        4: 'yellow',
        5: 'orange',
        6: 'magenta',
        7: 'cyan',
    }
    plt.figure(outname.replace('.pdf', ''))
    for ik in range(k):
        data_k = data[labels == ik]
        X = list(zip(*data_k))
        x, y = X[0], X[1]
        plt.scatter(x, y, c = colors[ik])
    plt.savefig(outname, format = 'pdf')


def create_model(k: int):
    """
    Create Sequential model for multiclass classifier
     k: number of clusters/classes
    """
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=2))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(k, activation='softmax'))
    return model


def plot_loss(history):
    """ Plot loss and val_loss """
    history_df = pd.DataFrame(history.history)
    plt.figure('loss')
    plt.plot(history_df['loss'], label='loss', c = 'black')
    plt.plot(history_df['val_loss'], label='val_loss', c = 'red')
    plt.legend()
    plt.savefig("loss.pdf")


def compile(model, optimizer = 'Adam'):
    """ Compile model """
    opt = optimizers.Adam(learning_rate = 0.001) if optimizer == 'Adam' else keras.optimizers.SGD(learning_rate=0.1)
    model.compile(
        loss='categorical_crossentropy',
        optimizer = opt,
        metrics = ['accuracy']
    )


def train(model, x_train, y_train):
    """ Train model """
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience = 5,
            monitor = 'val_loss',
            mode = 'min',
            restore_best_weights = True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            mode = 'min',
            verbose=0,
            save_best_only = True
        ),
    ]
    return model.fit(
        x_train,
        y_train,
        epochs = 300,
        validation_split = 0.2,
        callbacks = callbacks,
    )


def create_and_compile_model(k):
    """ Create and compile model """
    # Create model
    model = create_model(k)
    model.summary()

    # Compile model
    compile(model)
    
    return model


def normalize_data(data):
    """ Normalize data using MinMaxScaler """
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized


def main(k: int, train_model: bool):
    """
    k: number of clusters/classes
    train_model: train model or load previous trained model
    """
    if not train_model and not os.path.exists('best_model.h5'):
        print('ERROR: best_model.h5 file can not be found, run without the --loadBestModel flag for training a model first, exiting')
        sys.exit(1)
    
    # Create data
    data, labels = create_data(k, 100000)
    make_scatter_plot(data, labels, k, 'data.pdf')
    
    if train_model:
        model = create_and_compile_model(k)

    # Normalize data
    data_normalized = normalize_data(data)
    make_scatter_plot(data_normalized, labels, k, 'data_normalized.pdf')
    
    # Prepare labels
    labels = labels.reshape(-1, 1)
    labels = keras.utils.to_categorical(labels, num_classes=k)
     
    # Prepare train/test data
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size = 0.3, random_state=10)

    if train_model:
        # Train model
        history = train(model, x_train, y_train)

        # Plot loss
        plot_loss(history)

    # Load best model
    model = keras.models.load_model('best_model.h5')

    # Evaluate model in test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Loss in test data:', score[0])
    print('Accuracy in test data:', score[1])

    # Get predicted labels on full data
    predictions = model.predict(data_normalized)
    labels_predicted = np.argmax(predictions, axis=1)
    make_scatter_plot(data_normalized, labels_predicted, k, 'data_predicted.pdf')

    print('>>> ALL DONE <<<')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', '-k', type=int, action='store', default=3, help='Number of clusters/classes')
    parser.add_argument('--loadBestModel', action='store_true', default=False, help='No model will be trained and will load model from best_model.h5')
    args = parser.parse_args()
    main(args.k, not args.loadBestModel)