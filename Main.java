import IO.Dataset;
import IO.Output;
import Model.ActivationFunctions.ActivatorFunction;
import Model.Components.Layer;
import Model.Model;
import Model.ActivationFunctions.ReLuFunction;
import Model.ActivationFunctions.SigmoidFunction;

import java.util.ArrayList;
import java.util.List;

public class Main {
    private static int testIndex;
    private static int nOfHiddenPerceptrons = 2;
    private static ActivatorFunction hiddenLayerActivatorFunction = new ReLuFunction();
    private static ActivatorFunction outputLayerActivatorFunction = new SigmoidFunction();
    private static List<ActivatorFunction> functions = new ArrayList<>();
    private static Float alpha = 0.3F;
    private static Dataset dataset;

    public static void main(String[] args) {
        if (args.length != 2) {
            throw new IllegalArgumentException("Dataset path and label length should be indicated");
        }

        String datasetPath = args[0];
        int labelLength = Integer.parseInt(args[1]);

        dataset = new Dataset(datasetPath, labelLength);
        Output output = new Output();

        Model model = new Model(dataset, output);

        //testNumberOfHiddenPerceptrons(model, output);
        //testActivationFunctions(model, output);
        //testAlpha(model, output);
        //output.printFinalExecution();

        model.trainModel();
        model.testModel();

        output.generateOutputFiles();
    }

    public static void testNumberOfHiddenPerceptrons(Model model, Output output) {
        testIndex = 1;
        long bestTime = Long.MAX_VALUE;
        for (int nOfPerceptrons = 2; nOfPerceptrons < 9; nOfPerceptrons++) {
            Layer hiddenLayer = new Layer(nOfPerceptrons, model.getInputLayer(), hiddenLayerActivatorFunction);
            Layer outputLayer = new Layer(dataset.getLabelLength(), hiddenLayer, outputLayerActivatorFunction);
            model.setHiddenLayer(hiddenLayer);
            model.setOutputLayer(outputLayer);

            long time = testModel(model, output, "Testing Number of Hidden Perceptrons - Test #" + testIndex);

            if (time < bestTime) {
                nOfHiddenPerceptrons = nOfPerceptrons;
                bestTime = time;
            }
        }
        output.printTestResult("BEST NUMBER OF HIDDEN PERCEPTRON IS: " + nOfHiddenPerceptrons);
        model.updateLayers(nOfHiddenPerceptrons, hiddenLayerActivatorFunction, outputLayerActivatorFunction);
    }

    public static void testActivationFunctions(Model model, Output output) {
        testIndex = 1;

        functions.add(new SigmoidFunction());
        functions.add(new ReLuFunction());

        long bestTime = Long.MAX_VALUE;

        for (ActivatorFunction hiddenFunction : functions) {
            for (ActivatorFunction outputFunction : functions) {
                model.getHiddenLayer().setFunction(hiddenFunction);
                model.getOutputLayer().setFunction(outputFunction);

                long time = testModel(model, output, "Testing Activator Functions - Test #" + testIndex);
                if (time < bestTime) {
                    hiddenLayerActivatorFunction = hiddenFunction;
                    outputLayerActivatorFunction = outputFunction;
                    bestTime = time;
                }
            }
        }

        output.printTestResult("BEST ACTIVATOR FUNCTION FOR HIDDEN LAYER IS: " + hiddenLayerActivatorFunction
        + "\nBEST ACTIVATOR FUNCTION FOR OUTPUT LAYER IS: " + outputLayerActivatorFunction);
        model.updateLayers(nOfHiddenPerceptrons, hiddenLayerActivatorFunction, outputLayerActivatorFunction);
    }

    public static void testAlpha(Model model, Output output) {
        testIndex = 1;

        long bestTime = Long.MAX_VALUE;

        for (float currentAlpha = 0.1F; currentAlpha < 1.0F; currentAlpha += 0.1F) {
            model.setAlpha(currentAlpha);
            long time = testModel(model, output, "Testing Alphas - Test #" + testIndex);

            if (time < bestTime) {
                alpha = currentAlpha;
                bestTime = time;
            }
        }

        output.printTestResult("BEST ALPHA IS: " + alpha);
        model.setAlpha(alpha);
    }

    private static long testModel(Model model, Output output, String testName) {
        output.printTestHeader(testName);
        long time = model.trainModel();
        model.testModel();
        output.printTestSummary(model.getHiddenLayer(), model.getOutputLayer(), model.getAlpha(), time);
        testIndex++;

        return time;
    }
}
