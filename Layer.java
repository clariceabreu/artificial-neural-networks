import java.util.*;

public class Layer {
    //get
    private List<Perceptron> perceptrons;

    private ActivatorFunction function;

    //Pesos random
    public Layer(int numberOfPerceptrons, Layer previousLayer, ActivatorFunction function) {
        this.perceptrons = new ArrayList<>();
        this.function = function;
            
        for (int i = 0; i < numberOfPerceptrons; i++) {
            List<Perceptron> perceptronsFromPreviousLayer = new ArrayList<>();
            
            if (previousLayer != null) {
                perceptronsFromPreviousLayer = previousLayer.getPerceptrons();
            }

            Perceptron perceptron = new Perceptron(perceptronsFromPreviousLayer, this);

            this.perceptrons.add(perceptron);
        }
    }

    //Pesos fixos
    public Layer(int numberOfPerceptrons, Layer previousLayer, ActivatorFunction function, ArrayList<ArrayList<Float>> weights) {
        this.perceptrons = new ArrayList<>();
        this.function = function;

        for (int i = 0; i < numberOfPerceptrons; i++) {
            List<Perceptron> perceptronsFromPreviousLayer = new ArrayList<>();

            if (previousLayer != null) {
                perceptronsFromPreviousLayer = previousLayer.getPerceptrons();
            }

            Perceptron perceptron = new Perceptron(perceptronsFromPreviousLayer, this, weights.get(i));

            this.perceptrons.add(perceptron);
        }
    }

    public Float activate(Float signal) {
        return this.function.activate(signal);
    }

    public Float derived(Float signal) {
        return this.function.derived(signal);
    }

    public List<Perceptron> getPerceptrons() {
        return perceptrons;
    }
}