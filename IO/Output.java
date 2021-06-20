package IO;

import Model.Components.Layer;
import Model.Components.Perceptron;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Output {
    private PrintWriter initialParamsOutput;
    private PrintWriter initialWeightsOutput;
    private PrintWriter trainOutput;
    private PrintWriter finalWeightsOutput;
    private PrintWriter modelOutput;
    private PrintWriter trainErrorsOutput;
    private PrintWriter testSummaryOutput;

    private List<PrintWriter> allFiles;

    private static int correctResponses = 0;
    private static int wrongResponses = 0;

    public Output() {
        File outputsDir = new File("outputs");
        if (!outputsDir.exists()) {
            outputsDir.mkdir();
        }
        try {
            this.initialParamsOutput = new PrintWriter("outputs/initial_params.txt", "UTF-8");
            this.initialWeightsOutput = new PrintWriter("outputs/initial_weights.txt", "UTF-8");
            this.trainOutput = new PrintWriter("outputs/train_model.txt", "UTF-8");
            this.finalWeightsOutput = new PrintWriter("outputs/final_weights.txt", "UTF-8");
            this.modelOutput = new PrintWriter("outputs/model_output.txt", "UTF-8");
            this.trainErrorsOutput = new PrintWriter("outputs/train_errors.txt", "UTF-8");
            this.testSummaryOutput = new PrintWriter("outputs/tests_summary.txt", "UTF-8");
        } catch (IOException e) {
            System.out.println("An error occurred while creating output files");
            e.printStackTrace();
        }

        allFiles = new ArrayList<>();
        allFiles.add(initialParamsOutput);
        allFiles.add(initialWeightsOutput);
        allFiles.add(trainOutput);
        allFiles.add(finalWeightsOutput);
        allFiles.add(trainErrorsOutput);
        allFiles.add(modelOutput);
        allFiles.add(testSummaryOutput);
    }

    public void printInitialParams(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, Float alpha) {
        this.initialParamsOutput.println("Alpha: " + alpha);
        this.initialParamsOutput.println();

        this.initialParamsOutput.println("Number of perceptrons in the input layer: " + inputLayer.getPerceptrons().size());
        this.initialParamsOutput.println();

        this.initialParamsOutput.println("Number of perceptrons in the hidden layer: " + hiddenLayer.getPerceptrons().size());
        this.initialParamsOutput.println("Activator function in the hidden layer: " + hiddenLayer.getFunction().getFunctionName());
        this.initialParamsOutput.println();

        this.initialParamsOutput.println("Number of perceptrons in the output layer: " + outputLayer.getPerceptrons().size());
        this.initialParamsOutput.println("Activator function in the output layer: " + outputLayer.getFunction().getFunctionName());

        printWeights(hiddenLayer.getPerceptrons(), this.initialWeightsOutput, "hidden layer");
        printWeights(outputLayer.getPerceptrons(), this.initialWeightsOutput, "output layer");
    }

    public void printTrainStep(Layer hiddenLayer, Layer outputLayer, Float error, int epoch) {
        this.trainOutput.println("--------------------------------Epoch " + epoch + "--------------------------------");
        printWeights(hiddenLayer.getPerceptrons(), this.trainOutput, "hidden layer");
        printWeights(outputLayer.getPerceptrons(), this.trainOutput, "output layer");

        this.trainOutput.println("Mean square error: " + error);
        this.trainOutput.println();

        //Prints only the error in a separated file
        this.trainErrorsOutput.println(error);
    }

    public void printFinalWeights(Layer hiddenLayer, Layer outputLayer) {
        printWeights(hiddenLayer.getPerceptrons(), this.finalWeightsOutput, "hidden layer");
        printWeights(outputLayer.getPerceptrons(), this.finalWeightsOutput, "output layer");
    }

    public void printModelOutput(Layer outputLayer, DataVector test) {
        int[] inputsArray = new int[test.getInput().length];
        for (int i = 0; i < test.getInput().length; i++) {
            inputsArray[i] = (int) test.getInput()[i];
        }

        int[] expectedOutputArray = new int[test.getLabel().length];
        for (int i = 0; i < test.getLabel().length; i++) {
            expectedOutputArray[i] = Math.round(test.getLabel()[i]);
        }

        String input = Arrays.toString(inputsArray);
        String expectedOutput = Arrays.toString(expectedOutputArray);
        String rawOutput = Arrays.toString(outputLayer.getRawOutput());
        String output = Arrays.toString(outputLayer.getOutput());

        this.modelOutput.println("Inputs: " + input);
        this.modelOutput.println("Raw output is: " + rawOutput);
        this.modelOutput.println("Expected output: " + expectedOutput);
        this.modelOutput.println("Output: " + output);

        if(expectedOutput.equals(output)) {
            correctResponses++;
            this.modelOutput.println("Correct response ✓");
        } else {
            wrongResponses++;
            this.modelOutput.println("Wrong response ✖");
        }

        this.modelOutput.println();
    }

    public void printFinalResult(Float meanError) {
        this.modelOutput.println("---------------------------------------------------------------------------------------------------");
        this.modelOutput.println("Mean square error: " + meanError);
        this.modelOutput.println("Number of correct responses " + correctResponses + " out of " + (correctResponses + wrongResponses));
        this.modelOutput.println("---------------------------------------------------------------------------------------------------");
    }

    public void printTestSummary(Layer hiddenLayer, Layer outputLayer, Float alpha, long time) {
        this.testSummaryOutput.println("Number of hidden perceptrons: " + hiddenLayer.getPerceptrons().size());
        this.testSummaryOutput.println("Hidden layer activator function: " + hiddenLayer.getFunction().getFunctionName());
        this.testSummaryOutput.println("Output layer activator function: " + outputLayer.getFunction().getFunctionName());
        this.testSummaryOutput.println("Alpha: " + alpha);
        this.testSummaryOutput.println("Final mean square error: " + outputLayer.getMeanSquareError());
        this.testSummaryOutput.println("Time in ms: " + time);
        this.testSummaryOutput.println();
    }


    public void printTestResult(String result) {
        this.testSummaryOutput.println("---------------------------------------------------------------------------------------------------");
        this.testSummaryOutput.println();
        this.testSummaryOutput.println(result);
        this.testSummaryOutput.println();
    }

    public void printTestHeader(String testName) {
        allFiles.forEach(file -> file.println("-------------------------------" + testName + "-------------------------------"));
    }

    public void generateOutputFiles() {
        allFiles.forEach(PrintWriter::close);
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

    public static void setCorrectResponses(int correctResponses) {
        Output.correctResponses = correctResponses;
    }

    public static void setWrongResponses(int wrongResponses) {
        Output.wrongResponses = wrongResponses;
    }
}
