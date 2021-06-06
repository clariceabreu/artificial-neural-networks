import java.util.*;

public class Model
{
    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Fez bosta"); //TODO: trocar (acho que a Sara n vai gostar)
            return;
        }
        Model m = new Model(Integer.parseInt(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]));
        m.trainModel();
        m.testModel();
    }

    private final Float alpha = 0.3F;

    private DataSet dataset;

    private Layer inputLayer;
    private Layer hiddenLayer;
    private Layer outputLayer;

    public Model(int nOfInputPerceptrons, int nOfHiddenPerceptrons, int nOfOutputPerceptrons) {
        dataset = new DataSet("arquivoimaginarioquenaoexiste");

        initializeLayersWithRandomWeights(nOfInputPerceptrons, nOfHiddenPerceptrons, nOfOutputPerceptrons);
    }

    public void trainModel() {
        for (int epoca = 0; epoca < 100; epoca++) {
            for (Vector data : dataset.getTrainSet()){
                feedFoward(data);
                backPropagation(data);
                updateWeights();
            }
        }
    }

    public void testModel() {
        System.out.println("\033[1mResultados\033[m:");
        for (Vector test : dataset.getTestSet()) {
            feedFoward(test);

            for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
                System.out.println("\t\033[1;92m" + outputPerceptron.getOutputSignal() + "\033[m");
            }
        }
    }

    public void feedFoward(Vector data) {
        this.inputLayer.setOutput(data.input);
        this.hiddenLayer.calculateOutput();
        this.outputLayer.calculateOutput();
    }

    public void backPropagation(Vector data) {
        this.outputLayer.calculateErrorsFromLabel(alpha, data.label);
        this.hiddenLayer.calculateErrors(alpha);
    }

    public void updateWeights() {
        this.outputLayer.updateWeights();
        this.hiddenLayer.updateWeights();
    }

    private void initializeLayersWithRandomWeights(int nOfInputPerceptrons, int nOfHiddenPerceptrons, int nOfOutputPerceptrons) {
        System.out.println("Input (\033[1;93m" + nOfInputPerceptrons + "\033[m):");
        this.inputLayer = new Layer(nOfInputPerceptrons, null, null);
        System.out.println("Hidden (\033[1;93m" + nOfHiddenPerceptrons + "\033[m):");
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, new ReLuFunction());
        System.out.println("Output (\033[1;93m" + nOfOutputPerceptrons + "\033[m):");
        this.outputLayer = new Layer(nOfOutputPerceptrons, this.hiddenLayer, new SigmoidFunction());
    }

    private void initializeLayersWithFixedWeights(int nOfInputPerceptrons, int nOfHiddenPerceptrons, int nOfOutputPerceptrons) {
        ArrayList<ArrayList<Float>> hiddenLayerWeights = new ArrayList<>();
        ArrayList<Float> firstPerceptronOfHiddenLayer = new ArrayList<>();
        firstPerceptronOfHiddenLayer.add(0.53F);
        firstPerceptronOfHiddenLayer.add(0.97F);
        firstPerceptronOfHiddenLayer.add(0.43F);
        ArrayList<Float> secondPerceptronOfHiddenLayer = new ArrayList<>();
        secondPerceptronOfHiddenLayer.add(0.28F);
        secondPerceptronOfHiddenLayer.add(0.16F);
        secondPerceptronOfHiddenLayer.add(0.62F);
        hiddenLayerWeights.add(firstPerceptronOfHiddenLayer);
        hiddenLayerWeights.add(secondPerceptronOfHiddenLayer);

        ArrayList<ArrayList<Float>> outputLayerWeights = new ArrayList<>();
        ArrayList<Float> firstPerceptronOfOutputLayer = new ArrayList<>();
        firstPerceptronOfOutputLayer.add(0.17F);
        firstPerceptronOfOutputLayer.add(0.24F);
        firstPerceptronOfOutputLayer.add(0.43F);
        outputLayerWeights.add(firstPerceptronOfOutputLayer);

        this.inputLayer = new Layer(nOfInputPerceptrons, null, null);
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, new SigmoidFunction(), hiddenLayerWeights);
        this.outputLayer = new Layer(nOfOutputPerceptrons, this.hiddenLayer, new SigmoidFunction(), outputLayerWeights);
    }
}
