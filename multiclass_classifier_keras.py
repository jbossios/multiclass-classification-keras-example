import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import uniform, choice, normal
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras import Sequential
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Setting the seed with keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
keras.utils.set_random_seed(42)


def create_data(
        k: int,  # number of classes
        npoints: int,  # total number of points to be generated
        seed: int = 1  # seed for reproducibility
    ) -> tuple[np.array, np.array]:
    """
    Randomly create npoints data points classified into k classes
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    X = []  # collect data
    labels = []  # collect class labels
    
    # Randomly define a centroid for each class
    x_centers = {i: uniform(30, 70) for i in range(k)}
    y_centers = {i: uniform(0, 400000) for i in range(k)}
    
    # Generate npoints
    for ipoint in range(npoints):
        # Randomly assign this point to a class
        ik = choice(range(k))
        labels.append(ik)
        
        # Retrieve centroid for this class
        center_x = x_centers[ik]
        center_y = y_centers[ik]
        
        # Generate point
        X.append([
            normal(center_x, 3),
            normal(center_y, 15000)
        ])

    return np.array(X), np.array(labels)


def make_scatter_plot(
        data: np.array,  # 2D array
        labels: np.array,  # 1D array with class labels
        k: int,  # number of classes
        outname: str  # output name
    ) -> None:
    """ Make scatter plot"""
    
    # Protection
    assert k < 9, "Only up to 8 classes are supported"
    
    # Define colors for each class
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

    # Plot data from each class
    title = outname.split('.')[0]
    plt.figure(title)
    plt.title(title)
    for ik in range(k):  # loop over classes
        # Get data for class ik      
        data_k = data[labels == ik]

        # Unpack data
        X = list(zip(*data_k))
        x, y = X[0], X[1]

        # Make scatter plot and add x- and y-axis titles
        plt.scatter(x, y, c = colors[ik])

    # Save figure
    plt.savefig(outname)


def standardize_data(data: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
    """ Standardize data using StandardScaler """
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized


def create_model(
        k: int  # number of classes
    ) -> Sequential:
    """ Create Sequential model for multiclass classifier """
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=2))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(k, activation='softmax'))
    return model


def plot_loss(history) -> None:
    """ Plot loss and val_loss """
    history_df = pd.DataFrame(history.history)
    plt.figure('loss')
    plt.plot(history_df['loss'], label='loss', c='black')
    plt.plot(history_df['val_loss'], label='val_loss', c='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')


def compare_distributions(hists, k) -> None:
    """ Compare distributions """
    
    # Protection
    assert k < 9, "Only up to 8 classes are supported"

    # Define a color for every class
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
    
    # List of data categories
    data_categories = hists.keys()
    
    # Prepare the data
    data_dict = {}
    for i, (data_type, data) in enumerate(hists.items()):
        _, values = np.unique(data, return_counts=True)  # get counts for every class
        data_dict[data_type] = values
 
    # Create numerical axis
    x = np.arange(k)

    # Set the width of the bars
    width = 0.2

    # Create figure and axis
    fig, ax = plt.subplots()

    # Create bars
    for i, (data_type, counts) in enumerate(data_dict.items()):
        offset = width * i  # offset in the x-axis (different for each bar)
        bar = ax.bar(x + offset, counts, width=width, label=data_type, color=colors[i])
        ax.bar_label(bar)  # show numbers on top of bars

    # Label the axes
    ax.set_xlabel('Class')
    ax.set_ylabel('Counts')

    # Show category names
    ax.set_xticks(x + width, list(range(k)))

    # Add legends
    ax.legend()

    # Save figure
    fig.savefig('compare_distribution_of_classes_data.png')


def compile(model: Sequential, optimizer: str = 'Adam') -> None:
    """ Compile model """
    opt = optimizers.Adam(learning_rate=0.001) if optimizer == 'Adam' else optimizers.SGD(learning_rate=0.1)
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = opt,
        metrics = ['accuracy']
    )


def train(model, x_train, y_train, val_data):
    """ Train model """
    # Define callbacks
    callbacks = [
        # EarlyStopping: Stop training when a val_loss stops improving
        keras.callbacks.EarlyStopping(
            min_delta = 0.005,
            patience = 10,
            monitor = 'val_loss',
            mode = 'min',
            restore_best_weights = True
        ),
        # ModelCheckpoint: Save model w/ lowest val_loss
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor = 'val_loss',
            mode = 'min',
            verbose = 0,
            save_best_only = True
        ),
    ]
    return model.fit(
        x_train,
        y_train,
        epochs = 300,
        validation_data = val_data,
        callbacks = callbacks
    )


def create_and_compile_model(
        k: int  # number of classes
    ) -> Sequential:
    """ Create and compile model """
    # Create model
    model = create_model(k)
    model.summary()

    # Compile model
    compile(model)
    
    return model


def plot_confusion_matrix(cm):
    ax = sns.heatmap(cm, annot=True)
    ax.set_title('x-axis: true labels, y-axis: predicted labels')
    ax.get_figure().savefig('confusion_matrix.png')


def main(
        k: int,  # number of classes
        train_model: bool  # train model or load previous trained model
    ) -> None:
    
    # Protection
    if not train_model and not os.path.exists('best_model.h5'):
        print('ERROR: best_model.h5 file can not be found, run without the --loadBestModel flag for training a model first, exiting')
        sys.exit(1)
    
    # Create fake data
    data, labels = create_data(k=k, npoints=100)
    make_scatter_plot(data=data, labels=labels, k=k, outname='data.png')

    # Separate data into training, validation and testing data
    #   30% is used for testing
    #   20% of the remaining data is used for validation
    #   the rest is used for training
    x_train_tmp, X_test, y_train_tmp, y_test = train_test_split(data, labels, test_size=0.3, random_state=10)
    X_train, X_val, y_train, y_val = train_test_split(x_train_tmp, y_train_tmp, test_size=0.2, random_state=10)

    # Standardize all datasets
    data_dict = {
        'training': (X_train, y_train),
        'testing': (X_test, y_test),
        'validation': (X_val, y_val),
    }
    standardized_data_dict = {}
    for dataset_type, data_tuple in data_dict.items():
        data_standardized = standardize_data(data=data_tuple[0])
        standardized_data_dict[dataset_type] = (data_standardized, data_tuple[1])

    # Visualize standardized training dataset
    make_scatter_plot(
        data = standardized_data_dict['training'][0],
        labels = standardized_data_dict['training'][1],
        k = k,
        outname = 'standardized_training_data.png'
    )
    
    # Create and compile model (if training was requested)
    if train_model:
        model = create_and_compile_model(k=k)
    
    # Compare distribution of classes
    compare_distributions(
            {
                'all': labels,
                'training': y_train,
                'testing': y_test,
                'validation': y_val
            },
            k
        ) 

    # Train model or load best trained model (as requested)
    if train_model:
        history = train(
            model,
            standardized_data_dict['training'][0],  # training data
            standardized_data_dict['training'][1],  # training labels
            standardized_data_dict['validation']  # validation data and labels
        )
        plot_loss(history)  # plot loss vs epoch
    else:  # load best model    
        model = keras.models.load_model('best_model.h5')

    # Evaluate model in test data
    score = model.evaluate(
        standardized_data_dict['testing'][0],  # testing data
        standardized_data_dict['testing'][1],  # testing labels
        verbose = 0
    )
    print('Accuracy in test data:', score[1])

    # Get predicted labels on the test data
    predictions = model.predict(standardized_data_dict['testing'][0])
    labels_predicted = np.argmax(predictions, axis=1)  # find label with highest probability

    # Visualize test dataset with true labels
    make_scatter_plot(
        data = data_dict['testing'][0],  # data for testing
        labels = data_dict['testing'][1],  # labels for testing
        k = k,
        outname = 'test_data.png'
    )

    # Visualize test dataset with predicted labels
    make_scatter_plot(
        data = data_dict['testing'][0],  # data for testing
        labels = labels_predicted,  # predicted labels for test data
        k = k,
        outname = 'test_data_predicted_labels.png'
    )

    # Calculate and plot confusion matrix
    cm = confusion_matrix(data_dict['testing'][1], labels_predicted)
    plot_confusion_matrix(cm)

    print('>>> ALL DONE <<<')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', '-k', type=int, action='store', default=3, help='Number of classes')
    parser.add_argument('--loadBestModel', action='store_true', default=False, help='No model will be trained and will load model from best_model.h5')
    args = parser.parse_args()
    main(args.k, not args.loadBestModel)