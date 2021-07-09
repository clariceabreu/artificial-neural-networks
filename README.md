# ARTIFICIAL NEURAL NETWORKS

This program implements an artificial neural network using the Multilayer Perceptron (MLP) approach.
The application utilizes a supervised learning technique called backpropagation for training.

The program can ben run either using make (https://www.gnu.org/software/make/) or only java commands.

## Run with make
There are the following make commands available:
- `make`: compile the code
- `make run-charset`: run the code using the *dataset_chars_clean.csv* for both training and testing the model
- `make run-charset-noise`: run the code using *dataset_chars_clean.csv* for training and *dataset_chars_noise.csv* for testing the model
- `make run-charset-noise-20`: run the code using *dataset_chars_clean.csv* for training and *dataset_chars_noise_20.csv* for testing the model
- `make plot`: plot the graph with the mean square error values, if the early stop is set to true both the training errors and the validation errors are ploted, if it is not then only the training erros are ploted.
- `make clean`: remove the .class files from the code
- `make visualize-charset`: display the *dataset_chars_clean.csv* using colors to represent the chars

## Run with java commands
To run with java use the follwing commands:
1. `javac */*.java`: compile the code
2. `java Main [TRAINING DATASET RELATIVE PATH] [TESTING DATASET RELATIVE PATH] [LENGTH OF THE LABEL IN THE DATASET]'
- Example:
`java Main datasets/dataset_chars_clean.csv datasets/dataset_chars_noise.csv 7`
  The above command will run the code using *dataset_chars_clean.csv* for training and *dataset_chars_noise.csv* for testing the model
  
  
## Features 
### Test parameters
The application has methods to test the best parameters for the training.
There are test available to find out the best number of hidden perceptrons, the best activation function for the input and the output layer and the best alpha (learning tax).
To run the tests go to the Main class and uncomment lines 35 to 38. Be patient, they might take some time to complete.

### Early stop
To train the model two parameters can be changed to enable the training to stop before the maximum number of epochs (5000). These parameters are:
1. Boolean earlyStop: when set to true in each epoch the model will also be tested using the provided test dataset and the validation erros will be stored. When the validation error increases twice consecutively the model will stop training.
2. Float minError: when minError is greater than 0F the model will stop training as soon as the training mean square errors gets lower than the provided minError.
- Examples:
`model.trainModel(false, 0F)`: no early stop condition
`model.trainModel(true, 0F)`: training stops when validation errors start increasing
`model.trainModel(false, 0.01F)`: training stops when its mean square erros is lower than 0.01F
`model.trainModel(true, 0.01F)`: training stops when one of the two above cases happen

## Configurations
The default parameters are set as bellow:
- Alpha: `0.35`
- Number of perceptrons in hidden layer: `12`
- Activation function in hidden layer: `ReLu`
- Activation function in output layer: `Sigmoind`
- Max number of epochs: `5000`

Those parameters can be changed by cahnge their values in lines 21 to 25 in the Model class
PS: if the parameters tests are run the default values will be overwritten by the tests results.

## Outputs:
- `initial_parameters.txt`: has all the initial parameters used to train the model, i.e. alhpa, number of perceptron in hidden layer, activation functions
- `initial_weights.txt`: has all the ramdom initial weights used to start the training
- `train_model.txt`: has the weights for each epoch of the training
- `train_errors.txt`: has the mean square error of each epoch (used to plot the graph)
- `validation_errors.txt`: has the mean square error of each time the model is validated (when the earlyStop is set to true)
- `model_output.txt`: the final output of the model test.
- `model_confusion_matrix.txt`: the confusion matrix generated when the model is tested.
- `final_weights.txt`: has all the final weights for each synapses between percetrons
- `tests_summary.txt`: has the result of each iteration of the parameters tests
