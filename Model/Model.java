package Model;

import IO.Dataset;
import IO.DataVector;
import IO.Output;
import Model.ActivationFunctions.ActivatorFunction;
import Model.ActivationFunctions.ReLuFunction;
import Model.ActivationFunctions.SigmoidFunction;
import Model.Components.Layer;
import java.util.ArrayList;
import java.util.List;

public class Model {
    private Dataset dataset;
    private Output output;

    private Layer inputLayer;
    private Layer hiddenLayer;
    private Layer outputLayer;

    private Float alpha = 0.35F;
    private int nOfHiddenPerceptrons = 12;
    private ActivatorFunction hiddenLayerFunction = new ReLuFunction();
    private ActivatorFunction outputLayerFunction = new SigmoidFunction();
    private int maxEpochs = 5000;

    //Instantiates the model using the data specified when running the program
    public Model(Dataset dataset, Output output) {
        this.dataset = dataset;
        this.output = output;
    }

    //Trains the model
    public long trainModel(boolean earlyStop, Float minError) {        //Initial configurations
        this.initializeLayers();
        output.printInitialParams(inputLayer, hiddenLayer, outputLayer, alpha);
        long startTime = System.currentTimeMillis();

        int epoch = 0;
        Float meanError = 1F;
        List<Float> validationErrors = new ArrayList<>();
        List<Float> instantErrors = new ArrayList<>();

        //Iterates while stop conditions are not met (maximum number of epochs or the mean error)
        while (epoch <= maxEpochs && meanError > minError) {
            if (epoch % 10 == 0) {
                System.out.print("\033[2J\033[1;1H");
                System.out.print("Epoch: \033[1;93m" + epoch + "/" + maxEpochs + "\033[m");
            }

            //Iterates through every data in the dataset and does the feedforward, backpropagation and update weights steps
            for (DataVector data : dataset.getTrainSet()) {
                feedFoward(data);
                backPropagation(data);
                updateWeights();
                instantErrors.add(outputLayer.calculateInstantError(data));
            }

            //Calculates mean error to check early stop condition and increments number os epochs run
            meanError = outputLayer.calculateMeanSquareError(instantErrors);
            output.printTrainStep(hiddenLayer, outputLayer, meanError, epoch);

            //When the early stop param is true the model is validated to check
            //if it should stop the training early
            if (earlyStop) {
                Float validationError = testModel(true);
                validationErrors.add(validationError);

                //To check the early stop it is checked whereas the validation error
                //has increased in the last two epochs
                if (epoch > 2
                        && validationErrors.get(epoch) > validationErrors.get(epoch - 1)
                        && validationErrors.get(epoch - 1) > validationErrors.get(epoch - 2)) {
                    epoch = maxEpochs + 1;
                }
            }

            epoch++;
        }
        System.out.println();

        //Calculates the duration for the training
        long duration = System.currentTimeMillis() - startTime;
        output.printFinalWeights(hiddenLayer, outputLayer);

        return duration;
    }

    //Tests the model
    public Float testModel(boolean isValidation) {
        //Initial configuration of Output class attributes
        Output.setCorrectResponses(0);
        Output.setWrongResponses(0);
        Output.initializeConfusionMatrix(dataset.getLabelLength());

        List<Float> instantErrors = new ArrayList<>();

        //Iterates through every data in the test dataset using the feedforward method
        // (in the test, the model has already been trained and the weights have already been determined,
        // so only the feedforward step is run to get the outputs)
        for (DataVector test : dataset.getTestSet()) {
            feedFoward(test);
            instantErrors.add(outputLayer.calculateInstantError(test));

            if (!isValidation) output.printModelOutput(outputLayer, test);
        }
        output.printConfusionMatrix();

        Float meanError = outputLayer.calculateMeanSquareError(instantErrors);
        output.printTestError(meanError);

        if (!isValidation) {
            output.printFinalResult(meanError);
        }
        return meanError;
    }

    //Propagates the input signal through the next layers, applying the weights for each perceptron
    public void feedFoward(DataVector data) {
        this.inputLayer.setOutput(data.getInput());
        this.hiddenLayer.calculateOutput();
        this.outputLayer.calculateOutput();
    }

    //Propagates the error through to the previous layers, determining the weights and bias corrections
    public void backPropagation(DataVector data) {
        this.outputLayer.calculateErrorsFromLabel(alpha, data.getLabel());
        this.hiddenLayer.propagateError(alpha, outputLayer.getPerceptrons());
    }

    //Updates the weights and bias for each layer
    public void updateWeights() {
        this.outputLayer.updateWeights();
        this.hiddenLayer.updateWeights();
    }

    public Layer getHiddenLayer() { return this.hiddenLayer; }

    public Layer getOutputLayer() { return this.outputLayer; }

    public void setAlpha(Float alpha) { this.alpha = alpha; }

    public void setNOfHiddenPerceptrons(int nOfHiddenPerceptrons) {
        this.nOfHiddenPerceptrons = nOfHiddenPerceptrons;
    }

    public void setHiddenLayerFunction(ActivatorFunction hiddenLayerFunction) {
        this.hiddenLayerFunction = hiddenLayerFunction;
    }

    public void setOutputLayerFunction(ActivatorFunction outputLayerFunction) {
        this.outputLayerFunction = outputLayerFunction;
    }

    public Float getAlpha() { return alpha; }

    //Initializes each layer with the corresponding parameters
    public void initializeLayers() {
        this.inputLayer = new Layer(dataset.getInputLength(), null, null);
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, hiddenLayerFunction);
        this.outputLayer = new Layer(dataset.getLabelLength(), this.hiddenLayer, outputLayerFunction);
    }
}
