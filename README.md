# Example of a multiclass classifier with Keras

## Dependencies
```
numpy
pandas
matplotlib
keras from tensorflow
sklearn
```

## Run

In ```multiclass_classifier_keras.py```, you can find an example on how to train a multiclass classifier using Keras.

This example uses fake data generated randomly. A given number of clusters of people is created, each cluster have a different label, and we can learn that label based on the income and age of the people in a given cluster. We can choose the number of classes/clusters with the ```-k``` flag (if not provided, 3 clusters will be generated).

For generating the data, training the model and evaluating the peformance in testing data, run the following:


```
python multiclass_classifier_keras.py
```

This will save the best trained model to ```best_model.h5``` and will create four PDFs:

- ```data.pdf```: this is a scatter plot with the generated data
- ```data_normalized.pdf```:  this is the same as ```data.pdf``` but here the data was normalized
- ```data_predicted.pdf```: the same normalized data is plotted but the colors are based on the predicted labels (if the DNN works well it should look very similar to ```data_normalized.pdf```)
- ```loss.pdf```: this will have the loss and val_loss vs epoch 
