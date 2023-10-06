# Multiclass classifier with deep neural networks using the Keras API from TensorFlow

## Dependencies
```
numpy
pandas
matplotlib
seaborn
keras from tensorflow
sklearn
```

## Introduction

In ```multiclass_classifier_keras.py```, you can find an example on how to implement and train a multiclass classifier based on deep neural networks with Keras.

This example uses fake data, generated randomly. This data is characterized by two features and is classified into ```k``` labels. In this example, we will learn those labels.

## Run

For generating the data, training the model and evaluating the performance in testing data, run the following:


```
python multiclass_classifier_keras.py
```

Note:
- You can choose the number of classes with the ```-k``` flag (if not provided, 3 classes will be used)

The above script will save the best trained model to ```best_model.h5``` and will create five PNG images:

- ```data.png```: this is a scatter plot with the generated data, colored by the corresponding label
- ```standardized_training_data.png```:  this shows the data that will be used for training, which was already standardized
- ```test_data.png```:  this one shows the data that is used for testing
- ```test_data_predicted_labels.png```: the same test data is showed but data is colored based on the predicted labels (if the DNN works well it should look very similar to ```test_data.png```)
- ```loss.png```: this will have the loss and validation loss (val_loss) vs epoch
- ```compare_distribution_of_classes_data.png```: this compares the distribution of classes for each dataset type (training, test, validation, all=complete dataset)
- ```confusion_matrix.png```: as the name suggests, this will have the confusion matrix using the test data