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

        System.out.println("Input (\033[1;93m" + nOfInputPerceptrons + "\033[m):");
        this.inputLayer = new Layer(nOfInputPerceptrons, null, new SigmoidFunction());
        System.out.println("Hidden (\033[1;93m" + nOfHiddenPerceptrons + "\033[m):");
        this.hiddenLayer = new Layer(nOfHiddenPerceptrons, this.inputLayer, new SigmoidFunction());
        System.out.println("Output (\033[1;93m" + nOfOutputPerceptrons + "\033[m):");
        this.outputLayer = new Layer(nOfOutputPerceptrons, this.hiddenLayer, new SigmoidFunction());
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
        feedFoward(dataset.getTestSet().get(0));

        System.out.print("\033[1mResultados\033[m: ");
        for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
            System.out.println("\033[1;92m" + outputPerceptron.getOutputSignal() + "\033[m");
        }
    }


    public void feedFoward(Vector data) {
        List<Perceptron> perceptrons = this.inputLayer.getPerceptrons();

        //setando os valores de entrada dos neuronios na camada de entrada
        for (int i = 0; i < this.inputLayer.getPerceptrons().size(); i++) {
            perceptrons.get(i).setOutputSignal(data.input[i]);
        }

        for (Perceptron hiddenPerceptron : this.hiddenLayer.getPerceptrons()) {
            hiddenPerceptron.calculateOutput();
        }

        for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
            outputPerceptron.calculateOutput();
        }
    }

    public void backPropagation(Vector data) {
        for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
            Float outputSignal = outputPerceptron.getOutputSignal();
            Float inputSignal = outputPerceptron.getInputSignal();
            Float error = (data.label[0] - outputSignal)  * outputLayer.derived(inputSignal);
            outputPerceptron.calculateNewWeights(error, alpha);
        }

        for (Perceptron hiddenPerceptron : this.hiddenLayer.getPerceptrons()) {
            Float errorIn = 0.0F;

            for (Perceptron outputPerceptron : this.outputLayer.getPerceptrons()) {
                Float weight = outputPerceptron.getWeights().get(hiddenPerceptron);
                errorIn += weight * outputPerceptron.getError();
            }

            Float error = errorIn * hiddenLayer.derived(hiddenPerceptron.getInputSignal());

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
