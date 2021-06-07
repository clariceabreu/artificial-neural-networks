package Model;

import IO.Dataset;
import IO.DataVector;
import Model.ActivationFunctions.ReLuFunction;
import Model.ActivationFunctions.SigmoidFunction;
import Model.Components.Layer;
import java.util.ArrayList;

public class Model {
    private Dataset dataset;

    private Layer inputLayer;
    private Layer hiddenLayer;
    private Layer outputLayer;

    private Float alpha = 0.5F;
    private int nOfHiddenPerceptrons = 7;

    public Model(Dataset dataset) {
        this.dataset = dataset;
    }

    public void initializeModel() {
        System.out.println("-----------------------Initializing Model-----------------------");
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

        //printInitialParams(nOfInputPerceptrons, nOfHiddenPerceptrons, nOfOutputPerceptrons);
    }

    public void trainModel() {
        System.out.println("-------------------------Training Model-------------------------");
        for (int epoca = 0; epoca < 100; epoca++) {
            for (DataVector data : dataset.getTrainSet()){
                feedFoward(data);
                backPropagation(data);
                updateWeights();
                //printTrainingSteps(data);
            }
        }
    }

    public void testModel() {
        System.out.println("-------------------------Testlabeling Model-------------------------");
        for (DataVector test : dataset.getTestSet()) {
            feedFoward(test);
            printResult(test);
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

    private void printInitialParams(int nOfInputPerceptrons, int nOfHiddenPerceptrons, int nOfOutputPerceptrons) {
        System.out.println("Number of perceptrons in the input layer: " + nOfInputPerceptrons);
        System.out.println();

        System.out.println("Number of perceptrons in the hidden layer: " + nOfHiddenPerceptrons);
        System.out.println("Activator function in the hidden layer: " + this.hiddenLayer.getFunction().getFunctionName());
        System.out.println("Initial weights from input layer to hidden layer");
        this.hiddenLayer.printWeights();
        System.out.println();

        System.out.println("Number of perceptrons in the output layer: " + nOfOutputPerceptrons);
        System.out.println("Activator function in the output layer: " + this.outputLayer.getFunction().getFunctionName());
        System.out.println("Initial weights from hidden layer to output layer");
        this.outputLayer.printWeights();
        System.out.println();

        System.out.println("Alpha: " + this.alpha);
        System.out.println();
    }

    private void printTrainingSteps(DataVector data) {
        System.out.println("Weights from input layer to hidden layer");
        this.hiddenLayer.printWeights();
        System.out.println();

        System.out.println("Weights from hidden layer to output layer");
        this.outputLayer.printWeights();
        System.out.println();

        String expectedOutput = "[";
        for (int i = 0; i < data.getLabel().length; i++) {
            if (i > 0) expectedOutput += ',';
            expectedOutput += data.getLabel()[i];
        }
        expectedOutput += ']';

        System.out.println("Expected output: " + expectedOutput);
        System.out.println("Output: [" + String.join(",", this.outputLayer.getOutput()) + "]");
        System.out.println("Average error in output layer: " + this.outputLayer.getAverageError());
        System.out.println();
    }

    private void printResult(DataVector test) {
        String inputs = "[";
        for (int i = 0; i < test.getInput().length; i++) {
            if (i > 0) inputs += ',';
            inputs += test.getInput()[i];
        }
        inputs += ']';

        String outputs = "[";
        for (int i = 0; i < test.getLabel().length; i++) {
            if (i > 0) outputs += ',';
            outputs += test.getLabel()[i];
        }
        outputs += ']';

        System.out.println("Inputs: " + inputs);
        System.out.println("Expected output: " + outputs);
        System.out.println("Output: [" + String.join(",", this.outputLayer.getOutput()) + "]");
        System.out.println();
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
