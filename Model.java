import java.util.*;

public class Model {
    public Model(int nOfInputs, int nOfHiddens, int nOfOutputs) {
        this.NUMBER_OF_INPUT_PERCEPTRONS = nOfInputs;
        this.NUMBER_OF_HIDDEN_PERCEPTRONS = nOfHiddens;
        this.NUMBER_OF_OUTPUT_PERCEPTRONS = nOfOutputs;

        initializeDataset();

        System.out.println("Input (\033[1;93m" + nOfInputs + "\033[m):");
        this.inputLayer = new Layer(this.NUMBER_OF_INPUT_PERCEPTRONS, null, new SigmoidFunction());
        System.out.println("Hidden (\033[1;93m" + nOfHiddens + "\033[m):");
        this.hiddenLayer = new Layer(this.NUMBER_OF_HIDDEN_PERCEPTRONS, this.inputLayer, new SigmoidFunction());
        System.out.println("Output (\033[1;93m" + nOfOutputs + "\033[m):");
        this.outputLayer = new Layer(this.NUMBER_OF_OUTPUT_PERCEPTRONS, this.hiddenLayer, new SigmoidFunction());
    }

    private Layer inputLayer;

    private Layer hiddenLayer;

    private Layer outputLayer;

    //[entrada, entrada, target]
    private ArrayList<ArrayList<Float>> dataSet = new ArrayList<ArrayList<Float>>();

    private int NUMBER_OF_INPUT_PERCEPTRONS;
    private int NUMBER_OF_HIDDEN_PERCEPTRONS;
    private int NUMBER_OF_OUTPUT_PERCEPTRONS;
    
    private final Float alpha = 0.3F;

    private void initializeDataset() {
        ArrayList<Float> firstData = new ArrayList<>();
        firstData.add(1.0F);
        firstData.add(1.0F);
        firstData.add(1.0F);

        dataSet.add(firstData);

        ArrayList<Float> secondData = new ArrayList<>();
        secondData.add(0.0F);
        secondData.add(1.0F);
        secondData.add(0.0F);

        dataSet.add(secondData);

        ArrayList<Float> thirdData = new ArrayList<>();
        thirdData.add(1.0F);
        thirdData.add(0.0F);
        thirdData.add(0.0F);

        dataSet.add(thirdData);

        ArrayList<Float> fourthData = new ArrayList<>();
        fourthData.add(0.0F);
        fourthData.add(0.0F);
        fourthData.add(0.0F);

        dataSet.add(fourthData);
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Fez bosta"); //TODO: trocar (acho que a Sara n vai gostar)
            return;
        }
        Model m = new Model(Integer.parseInt(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]));
        m.trainModel();
        
        List<Float> testData = new ArrayList<>();
        testData.add(1.0F);
        testData.add(1.0F);
        m.testModel(testData);
    }

    public void trainModel() {
        for (int epoca = 0; epoca < 100; epoca++) {
            for (int i = 0; i < dataSet.size(); i++){
                feedFoward(this.dataSet.get(i));
                backPropagation(dataSet.get(i).get(2));
                updateWeights();
            }
        }
    }

    public void testModel(List<Float> inputSignals) {
        feedFoward(inputSignals);

        System.out.print"\033[1mResultados\033[m: ");
        for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
            System.out.println("\033[1;92m" + outputPerceptron.getOutputSignal() + "\033[m");
        }
    }


    public void feedFoward(List<Float> inputSignals) {
        List<Perceptron> perceptrons = this.inputLayer.getPerceptrons();

        //setando os valores de entrada dos neuronios na camada de entrada
        for (int i = 0; i < this.NUMBER_OF_INPUT_PERCEPTRONS; i++) {
            perceptrons.get(i).setOutputSignal(inputSignals.get(i));
        }

        for (Perceptron hiddenPerceptron : this.hiddenLayer.getPerceptrons()) {
            hiddenPerceptron.calculateOutput();
        }

        for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
            outputPerceptron.calculateOutput();
        }
    }

    public void backPropagation(Float target) {
        for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
            Float outputSignal = outputPerceptron.getOutputSignal();
            Float inputSignal = outputPerceptron.getInputSignal();
            Float error = (target - outputSignal)  * outputLayer.derived(inputSignal);
            outputPerceptron.calculateNewWeights(error, alpha);
        }

        for (Perceptron hiddenPerceptron : this.hiddenLayer.getPerceptrons()) {
            Float errorIn = 0.0f;
            
            for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
                Float weight = outputPerceptron.getWeights().get(hiddenPerceptron);
                errorIn += weight * outputPerceptron.getError(); //FIXME: n sei de ta certo
            }

            Float error = errorIn * hiddenLayer.derived(hiddenPerceptron.getOutputSignal());

            hiddenPerceptron.calculateNewWeights(error, alpha);
        }

    }

    public void updateWeights() {
        for (Perceptron perceptron : this.outputLayer.getPerceptrons()) {
            perceptron.updateWeights();
        }

        for (Perceptron perceptron : this.hiddenLayer.getPerceptrons()) {
            perceptron.updateWeights();
        }
    }
}