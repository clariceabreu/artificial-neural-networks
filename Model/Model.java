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

    public Model(Dataset dataset, Output output) {
        this.dataset = dataset;
        this.output = output;
    }

    public long trainModel(boolean earlyStop, Float minError) {
        this.initializeLayers();
        output.printInitialParams(inputLayer, hiddenLayer, outputLayer, alpha);
        long startTime = System.currentTimeMillis();

        int epoch = 0;
        Float meanError = 1F;
        List<Float> validationErrors = new ArrayList<>();
        List<Float> instantErrors = new ArrayList<>();

        while (epoch <= maxEpochs && meanError > minError) {
            if (epoch % 10 == 0) {
                System.out.print("\033[2J\033[1;1H");
                System.out.print("Epoch: \033[1;93m" + epoch + "/" + maxEpochs + "\033[m");
            }
            for (DataVector data : dataset.getTrainSet()) {
                feedFoward(data);
                backPropagation(data);
                updateWeights();
                instantErrors.add(outputLayer.calculateInstantError(data));
            }

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

        long duration = System.currentTimeMillis() - startTime;
        output.printFinalWeights(hiddenLayer, outputLayer);

        return duration;
    }

    public Float testModel(boolean isValidation) {
        Output.setCorrectResponses(0);
        Output.setWrongResponses(0);

        List<Float> instantErrors = new ArrayList<>();

        for (DataVector test : dataset.getTestSet()) {
            feedFoward(test);
            instantErrors.add(outputLayer.calculateInstantError(test));

            if (!isValidation) output.printModelOutput(outputLayer, test);
        }

        Float meanError = outputLayer.calculateMeanSquareError(instantErrors);
        output.printTestError(meanError);

        if (!isValidation) {
            output.printFinalResult(meanError);
        }
        return meanError;
    }

    public void feedFoward(DataVector data) {
        this.inputLayer.setOutput(data.getInput());
        this.hiddenLayer.calculateOutput();
        this.outputLayer.calculateOutput();
    }

    public void backPropagation(DataVector data) {
        this.outputLayer.calculateErrorsFromLabel(alpha, data.getLabel());
        this.hiddenLayer.propagateError(alpha, outputLayer.getPerceptrons());
    }

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

    public void initializeLayers() {
        this.inputLayer = new Layer(dataset.getInputLength(), null, null);
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, hiddenLayerFunction);
        this.outputLayer = new Layer(dataset.getLabelLength(), this.hiddenLayer, outputLayerFunction);
    }
}
