package Model;

import IO.Dataset;
import IO.DataVector;
import IO.Output;
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

    private Float alpha = 0.2F;
    private int nOfHiddenPerceptrons = 2;
    private final int maxEpochs = 5000;

    public Model(Dataset dataset) {
        this.dataset = dataset;
        this.output = new Output();
    }

    public void initializeModel() {
        int nOfInputPerceptrons = dataset.getInputLength();
        int nOfOutputPerceptrons = dataset.getLabelLength();

        if (this.inputLayer == null) {
            this.inputLayer = new Layer(nOfInputPerceptrons, null, null);
        }
        if (this.hiddenLayer == null) {
            this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, new ReLuFunction());
        }

        if (this.outputLayer == null) {
            this.outputLayer = new Layer(nOfOutputPerceptrons, this.hiddenLayer, new SigmoidFunction());
        }

        output.generateInitialParamsOutput(inputLayer, hiddenLayer, outputLayer, alpha);
    }

    public void trainModel() {
        List<Float> instantErrors = new ArrayList<>();
        int epoch = 1;

        while (epoch < maxEpochs && outputLayer.getMeanSquareError(instantErrors) > 0.01F) {
            for (DataVector data : dataset.getTrainSet()){
                feedFoward(data);
                backPropagation(data);
                updateWeights();
                instantErrors.add(outputLayer.getInstantError(data));
            }

            Float meanError = outputLayer.getMeanSquareError(instantErrors);
            output.printTrainStep(hiddenLayer, outputLayer, meanError, epoch);
            output.printError(meanError);
            epoch++;
        }

        output.generateTrainOutput();
        output.generateErrorOutput();
    }

    public void testModel() {
        for (DataVector test : dataset.getTestSet()) {
            feedFoward(test);
            output.printTestResult(outputLayer, test);
        }

        output.generateTestOutput();
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

    public void setAlpha(Float alpha) {
        this.alpha = alpha;
    }

    public void setNumberOfHiddenPerceptrons(int nOfHiddenPerceptrons) {
        this.nOfHiddenPerceptrons = nOfHiddenPerceptrons;
    }

    public Layer getHiddenLayer() {
        return hiddenLayer;
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }

    public void initializeLayersWithFixedWeights(int nOfInputPerceptrons, int nOfHiddenPerceptrons, int nOfOutputPerceptrons) {
        ArrayList<ArrayList<Float>> hiddenLayerWeights = new ArrayList<>();
        ArrayList<Float> firstPerceptronOfHiddenLayer = new ArrayList<>();
        firstPerceptronOfHiddenLayer.add(0.1F);
        firstPerceptronOfHiddenLayer.add(-0.1F);
        firstPerceptronOfHiddenLayer.add(-0.1F);

        ArrayList<Float> secondPerceptronOfHiddenLayer = new ArrayList<>();
        secondPerceptronOfHiddenLayer.add(0.1F);
        secondPerceptronOfHiddenLayer.add(0.1F);
        secondPerceptronOfHiddenLayer.add(-0.1F);

        ArrayList<Float> thirdPerceptronOfHiddenLayer = new ArrayList<>();
        thirdPerceptronOfHiddenLayer.add(-0.1F);
        thirdPerceptronOfHiddenLayer.add(-0.1F);
        thirdPerceptronOfHiddenLayer.add(0.1F);

        hiddenLayerWeights.add(firstPerceptronOfHiddenLayer);
        hiddenLayerWeights.add(secondPerceptronOfHiddenLayer);
        hiddenLayerWeights.add(thirdPerceptronOfHiddenLayer);

        ArrayList<ArrayList<Float>> outputLayerWeights = new ArrayList<>();
        ArrayList<Float> firstPerceptronOfOutputLayer = new ArrayList<>();
        firstPerceptronOfOutputLayer.add(0.1F);
        firstPerceptronOfOutputLayer.add(0.0F);
        firstPerceptronOfOutputLayer.add(0.1F);
        firstPerceptronOfOutputLayer.add(-0.1F);

        ArrayList<Float> secondPerceptronOfOutputLayer = new ArrayList<>();
        secondPerceptronOfOutputLayer.add(-0.1F);
        secondPerceptronOfOutputLayer.add(0.1F);
        secondPerceptronOfOutputLayer.add(-0.1F);
        secondPerceptronOfOutputLayer.add(0.1F);

        outputLayerWeights.add(firstPerceptronOfOutputLayer);
        outputLayerWeights.add(secondPerceptronOfOutputLayer);

        this.inputLayer = new Layer(nOfInputPerceptrons, null, null);
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, new SigmoidFunction(), hiddenLayerWeights);
        this.outputLayer = new Layer(nOfOutputPerceptrons, this.hiddenLayer, new SigmoidFunction(), outputLayerWeights);
    }
}
