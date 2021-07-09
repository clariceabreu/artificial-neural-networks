import IO.Dataset;
import IO.Output;
import Model.ActivationFunctions.ActivatorFunction;
import Model.Model;
import Model.ActivationFunctions.ReLuFunction;
import Model.ActivationFunctions.SigmoidFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static int nOfHiddenPerceptrons = 12;
    private static ActivatorFunction hiddenLayerActivatorFunction = new ReLuFunction();
    private static ActivatorFunction outputLayerActivatorFunction = new SigmoidFunction();
    private static List<ActivatorFunction> functions = new ArrayList<>();
    private static Float alpha = 0.35F;
    private static Dataset dataset;

    public static void main(String[] args) {
        if (args.length != 3) {
            throw new IllegalArgumentException("Train dataset path, test dataset path and label length should be indicated");
        }

        String trainDatasetPath = args[0];
        String testDatasetPath = args[1];
        int labelLength = Integer.parseInt(args[2]);

        dataset = new Dataset(trainDatasetPath, testDatasetPath, labelLength);
        Output output = new Output();

        Model model = new Model(dataset, output);

        //Uncomment the bellow lines to run tests to find the best parameters
        //testNumberOfHiddenPerceptrons(model, output);
        //testActivationFunctions(model, output);
        //testAlpha(model, output);
        //output.printFinalExecution();

        enterAltTermBuffer();

        model.trainModel(true, 0.01F);
        model.testModel(false);

        leaveAltTermBuffer();

        output.generateOutputFiles();
    }

    private static void testNumberOfHiddenPerceptrons(Model model, Output output) {
        long bestTime = Long.MAX_VALUE;
        for (int nOfPerceptrons = 2; nOfPerceptrons <= 63; nOfPerceptrons++) {
            System.out.println("Testing with " + nOfPerceptrons + " perceptrons in hidden layer");
            model.setNOfHiddenPerceptrons(nOfPerceptrons);
            long time = trainModel(model, output, "Testing Number of Hidden Perceptrons - #" + nOfPerceptrons);

            if (time < bestTime) {
                nOfHiddenPerceptrons = nOfPerceptrons;
                bestTime = time;
            }
        }

        model.setNOfHiddenPerceptrons(nOfHiddenPerceptrons);
        output.printTestResult("BEST NUMBER OF HIDDEN PERCEPTRON IS: " + nOfHiddenPerceptrons);
    }

    private static void testActivationFunctions(Model model, Output output) {
        functions.add(new SigmoidFunction());
        functions.add(new ReLuFunction());

        long bestTime = Long.MAX_VALUE;

        for (ActivatorFunction hiddenFunction : functions) {
            for (ActivatorFunction outputFunction : functions) {
                System.out.println("Testing with " + hiddenFunction.getFunctionName() + " function in hidden layer and "
                        + outputFunction.getFunctionName() + " in output layer");


                model.setHiddenLayerFunction(hiddenFunction);
                model.setOutputLayerFunction(outputFunction);

                long time = trainModel(model, output, "Testing Activator Functions - Hidden: " + hiddenFunction.getFunctionName() + " - Output: " + outputFunction.getFunctionName());
                if (time < bestTime) {
                    hiddenLayerActivatorFunction = hiddenFunction;
                    outputLayerActivatorFunction = outputFunction;
                    bestTime = time;
                }
            }
        }

        model.setHiddenLayerFunction(hiddenLayerActivatorFunction);
        model.setOutputLayerFunction(outputLayerActivatorFunction);
        output.printTestResult("BEST ACTIVATOR FUNCTION FOR HIDDEN LAYER IS: " + hiddenLayerActivatorFunction.getFunctionName()
        + "\nBEST ACTIVATOR FUNCTION FOR OUTPUT LAYER IS: " + outputLayerActivatorFunction.getFunctionName());
    }

    private static void testAlpha(Model model, Output output) {
        long bestTime = Long.MAX_VALUE;

        for (float currentAlpha = 0.01F; currentAlpha < 1.0F; currentAlpha += 0.05F) {
            System.out.println("Testing with alpha " + currentAlpha);
            model.setAlpha(currentAlpha);
            long time = trainModel(model, output, "Testing Alphas - #" + currentAlpha);

            if (time < bestTime) {
                alpha = currentAlpha;
                bestTime = time;
            }
        }

        output.printTestResult("BEST ALPHA IS: " + alpha);
        model.setAlpha(alpha);
    }

    private static long trainModel(Model model, Output output, String testName) {
        output.printTestHeader(testName);
        long time = model.trainModel(true, 0.1F);
        output.printTestSummary(model.getHiddenLayer(), model.getOutputLayer(), model.getAlpha(), time);

        return time;
    }

    private static void enterAltTermBuffer() {
        System.out.print("\033[?1049h\033[?25l");
    }

    private static void leaveAltTermBuffer() {
        System.out.print("\nPress \033[1;93m[enter]\033[m to exit");
        Scanner s = new Scanner(System.in);
        s.nextLine();
        System.out.print("\033[?1049l\033[?25h");
    }
}
