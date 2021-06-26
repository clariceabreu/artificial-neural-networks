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

    public long trainModel() {
        this.initializeLayers();
        output.printInitialParams(inputLayer, hiddenLayer, outputLayer, alpha);
        long startTime = System.currentTimeMillis();

        int epoch = 1;
        Float meanError = 1F;
        List<Float> instantErrors = new ArrayList<>();

        //TO DO: trabalhar melhor a condição de parada
        while (epoch <= maxEpochs && meanError > 0.01F) {
            if (epoch % 10 == 0) System.out.print("\rEpoch: " + epoch + "/" + maxEpochs);
            for (DataVector data : dataset.getTrainSet()) {
                feedFoward(data);
                backPropagation(data);
                updateWeights();
                instantErrors.add(outputLayer.calculateInstantError(data));
            }

            meanError = outputLayer.calculateMeanSquareError(instantErrors);
            output.printTrainStep(hiddenLayer, outputLayer, meanError, epoch);
            epoch++;
        }
        System.out.println();

        long duration = System.currentTimeMillis() - startTime;
        output.printFinalWeights(hiddenLayer, outputLayer);

        return duration;
    }

    public void testModel() {
        Output.setCorrectResponses(0);
        Output.setWrongResponses(0);

        List<Float> instantErrors = new ArrayList<>();

        for (DataVector test : dataset.getTestSet()) {
            feedFoward(test);
            instantErrors.add(outputLayer.calculateInstantError(test));
            output.printModelOutput(outputLayer, test);
        }

        Float meanError = outputLayer.calculateMeanSquareError(instantErrors);
        output.printTestError(meanError);
        output.printFinalResult(meanError);
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
