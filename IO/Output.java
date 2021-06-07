package IO;

import Model.Components.Layer;
import Model.Components.Perceptron;

import java.io.*;
import java.util.List;

public class Output {
    private PrintWriter initialParamsOutput;
    private PrintWriter trainOutput;
    private PrintWriter testOutput;
    private PrintWriter errorsOutput;

    public Output() {
        try {
            this.initialParamsOutput = new PrintWriter("outputs/initial_params.txt", "UTF-8");
            this.trainOutput = new PrintWriter("outputs/train_model.txt", "UTF-8");
            this.testOutput = new PrintWriter("outputs/test_model.txt", "UTF-8");
            this.errorsOutput = new PrintWriter("outputs/errors.txt", "UTF-8");
        } catch (IOException e) {
            System.out.println("An error occurred while creating output files");
            e.printStackTrace();
        }
    }

    public void generateInitialParamsOutput(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, Float alpha) {
        this.initialParamsOutput.println("Alpha: " + alpha);
        this.initialParamsOutput.println();

        this.initialParamsOutput.println("Number of perceptrons in the input layer: " + inputLayer.getPerceptrons().size());
        this.initialParamsOutput.println();

        this.initialParamsOutput.println("Number of perceptrons in the hidden layer: " + hiddenLayer.getPerceptrons().size());
        this.initialParamsOutput.println("Activator function in the hidden layer: " + hiddenLayer.getFunction().getFunctionName());
        printWeights(hiddenLayer.getPerceptrons(), this.initialParamsOutput, "hidden layer");
        this.initialParamsOutput.println();

        this.initialParamsOutput.println("Number of perceptrons in the output layer: " + outputLayer.getPerceptrons().size());
        this.initialParamsOutput.println("Activator function in the output layer: " + outputLayer.getFunction().getFunctionName());
        printWeights(outputLayer.getPerceptrons(), this.initialParamsOutput, "output layer");
        this.initialParamsOutput.println();

        this.initialParamsOutput.close();
    }

    public void printTrainStep(Layer hiddenLayer, Layer outputLayer, Float error, int epoch) {
        this.trainOutput.println("--------------------------------Epoch " + epoch + "--------------------------------");
        printWeights(hiddenLayer.getPerceptrons(), this.trainOutput, "hidden layer");
        printWeights(outputLayer.getPerceptrons(), this.trainOutput, "output layer");

        this.trainOutput.println("Mean square error: " + error);
        this.trainOutput.println();
    }

    public void printError(Float error) {
        this.errorsOutput.println(error);
    }

    public void printTestResult(Layer outputLayer, DataVector test) {
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

        this.testOutput.println("Inputs: " + inputs);
        this.testOutput.println("Expected output: " + outputs);
        this.testOutput.println("Output: [" + String.join(",", outputLayer.getOutput()) + "]");
        this.testOutput.println();
    }

    public void generateTrainOutput() {
        this.trainOutput.close();
    }

    public void generateErrorOutput() {
        this.errorsOutput.close();
    }

    public void generateTestOutput() {
        this.testOutput.close();
    }

    private void printWeights(List<Perceptron> perceptrons, PrintWriter out, String layer) {
        for (int i = 0; i < perceptrons.size(); i++) {
            out.write("Input weights for perceptron " + (i + 1) + " of " + layer + ": ");
            int weightIndex = 0;
            for (Float weight : perceptrons.get(i).getWeights().values()) {
                if (weightIndex == 0) {
                    out.println(weight);
                } else {
                    out.println("                                                " + weight);
                }
                weightIndex++;
            }
            out.println("                                                " + perceptrons.get(i).getBiasWeight() + " (bias)");
            out.println();
        }
    }
}
