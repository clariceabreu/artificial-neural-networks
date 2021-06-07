import IO.Dataset;
import IO.Output;
import Model.Components.Layer;
import Model.Model;
import Model.ActivationFunctions.ReLuFunction;
import Model.ActivationFunctions.SigmoidFunction;

public class Main {
    private static int testIndex = 1;
    private static int nOfInputPerceptrons;
    private static int nOfOutputPerceptrons;


    public static void main(String[] args) {
        if (args.length != 2) {
            throw new IllegalArgumentException("Dataset path and label length should be indicated");
        }

        String datasetPath = args[0];
        int labelLength = Integer.parseInt(args[1]);

        Dataset dataset = new Dataset(datasetPath, labelLength);
        nOfInputPerceptrons = dataset.getInputLength();
        nOfOutputPerceptrons = dataset.getLabelLength();

        Output output = new Output();

        Model model = new Model(dataset, output);
        model.trainModel();
        model.testModel();

        //testNumberOfHiddenPerceptrons(model, output);
        //testActivationFunctions(model, output);
        //testAlpha(model, output);

        output.generateOutputFiles();
    }

    public static void testNumberOfHiddenPerceptrons(Model model, Output output) {
        for (int nOfHiddenPerceptrons = 5; nOfHiddenPerceptrons < 10; nOfHiddenPerceptrons++) {
            Layer inputLayer = new Layer(nOfInputPerceptrons, null, null);
            Layer hiddenLayer = new Layer(nOfHiddenPerceptrons, inputLayer, new ReLuFunction());
            Layer outputLayer = new Layer(nOfOutputPerceptrons, hiddenLayer, new SigmoidFunction());

            model.setInputLayer(inputLayer);
            model.setHiddenLayer(hiddenLayer);
            model.setOutputLayer(outputLayer);

            testModel(model, output);
        }
    }

    public static void testActivationFunctions(Model model, Output output) {
        model.getHiddenLayer().setFunction(new ReLuFunction());
        model.getOutputLayer().setFunction(new ReLuFunction());
        testModel(model, output);


        model.getHiddenLayer().setFunction(new ReLuFunction());
        model.getOutputLayer().setFunction(new SigmoidFunction());
        testModel(model, output);

        model.getHiddenLayer().setFunction(new SigmoidFunction());
        model.getOutputLayer().setFunction(new ReLuFunction());
        testModel(model, output);

        model.getHiddenLayer().setFunction(new SigmoidFunction());
        model.getOutputLayer().setFunction(new SigmoidFunction());
        testModel(model, output);
    }

    public static void testAlpha(Model model, Output output) {
        for (float alpha = 0.1F; alpha < 1.0F; alpha += 0.1F) {
            model.setAlpha(alpha);
            testModel(model, output);
        }
    }

    private static void testModel(Model model, Output output) {
        output.printTestNumber(testIndex);
        model.randomizePerceptronsWeights();
        model.trainModel();
        model.testModel();
        testIndex++;
    }
}