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

    private Float alpha = 0.02F;
    private final int nOfHiddenPerceptrons = 4;
    private final int maxEpochs = 3000;

    public Model(Dataset dataset, Output output) {
        this.dataset = dataset;
        this.output = output;

        int nOfInputPerceptrons = dataset.getInputLength();
        int nOfOutputPerceptrons = dataset.getLabelLength();

        this.inputLayer = new Layer(nOfInputPerceptrons, null, null);
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, new ReLuFunction());
        this.outputLayer = new Layer(nOfOutputPerceptrons, this.hiddenLayer, new SigmoidFunction());
    }

    public long trainModel() {
        output.printInitialParams(inputLayer, hiddenLayer, outputLayer, alpha);
        long startTime = System.currentTimeMillis();

        int epoch = 1;
        Float meanError = 1F;

        while (epoch < maxEpochs /* && meanError > 0.01F */) {
            for (int i = 0; i < dataset.getTrainSet().size(); i++) {
                DataVector data = dataset.getTrainSet().get(i);
                feedFoward(data);
                backPropagation(data);
                updateWeights();
                outputLayer.calculateInstantError(data, i);
            }

            meanError = outputLayer.getMeanSquareError();
            output.printTrainStep(hiddenLayer, outputLayer, meanError, epoch);
            epoch++;
        }

        long duration = System.currentTimeMillis() - startTime;

        return duration;
    }

    public void testModel() {
        for (DataVector test : dataset.getTestSet()) {
            feedFoward(test);
            output.printTestResult(outputLayer, test);
        }
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

    public Layer getHiddenLayer() { return hiddenLayer; }

    public Layer getOutputLayer() { return outputLayer; }

    public void setInputLayer(Layer inputLayer) { this.inputLayer = inputLayer; }

    public void setHiddenLayer(Layer hiddenLayer) { this.hiddenLayer = hiddenLayer; }

    public void setOutputLayer(Layer outputLayer) { this.outputLayer = outputLayer; }

    public void setAlpha(Float alpha) { this.alpha = alpha; }

    public Float getAlpha() { return alpha; }

    public void randomizePerceptronsWeights() {
        this.hiddenLayer.randomizePerceptronsWeights();
        this.outputLayer.randomizePerceptronsWeights();
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
