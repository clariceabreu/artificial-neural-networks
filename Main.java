import IO.Dataset;
import Model.Model;
import Model.ActivationFunctions.ReLuFunction;
import Model.ActivationFunctions.SigmoidFunction;

public class Main {
    public static void main(String[] args) {
        if (args.length != 2) {
            throw new IllegalArgumentException("Dataset path and label length should be indicated");
        }

        String datasetPath = args[0];
        int labelLength = Integer.parseInt(args[1]);

        Dataset dataset = new Dataset(datasetPath, labelLength);

        Model model = new Model(dataset);
        model.initializeModel();
        model.trainModel();
        model.testModel();

        //testNumberOfHiddenPerceptrons(model);
        //testActivationFunctions(model);
        //testAlpha(model);
    }

    public static void testNumberOfHiddenPerceptrons(Model model) {
        System.out.println("--------Test with different number of hidden perceptrons--------");
        for (int i = 5; i < 10; i++) {
            model.setNumberOfHiddenPerceptrons(i);
            model.initializeModel();
            model.trainModel();
            model.testModel();
        }
    }

    public static void testActivationFunctions(Model model) {
        System.out.println("-------------Test with different activator functions-------------");
        model.getHiddenLayer().setFunction(new ReLuFunction());
        model.getOutputLayer().setFunction(new ReLuFunction());
        model.initializeModel();
        model.trainModel();
        model.testModel();


        model.getHiddenLayer().setFunction(new ReLuFunction());
        model.getOutputLayer().setFunction(new SigmoidFunction());
        model.initializeModel();
        model.trainModel();
        model.testModel();

        model.getHiddenLayer().setFunction(new SigmoidFunction());
        model.getOutputLayer().setFunction(new ReLuFunction());
        model.initializeModel();
        model.trainModel();
        model.testModel();

        model.getHiddenLayer().setFunction(new SigmoidFunction());
        model.getOutputLayer().setFunction(new SigmoidFunction());
        model.initializeModel();
        model.trainModel();
        model.testModel();
    }

    public static void testAlpha(Model model) {
        System.out.println("-------------------------Test with alphas-------------------------");
        for (float alpha = 0.1F; alpha < 1.0F; alpha += 0.1F) {
            model.setAlpha(alpha);
            model.initializeModel();
            model.trainModel();
            model.testModel();
        }
    }
}