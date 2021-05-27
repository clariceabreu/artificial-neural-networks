import java.util.*;

public class Perceptron {
    //get
    private Float inputSignal;

    //get
    private Float outputSignal;

    //get & set
    private HashMap<Perceptron, Float> weights;
    private Float biasWeight;

    //get & set
    private HashMap<Perceptron, Float> newWeights;

    //get
    private Float error;

    private Layer layer;


    public Perceptron(List<Perceptron> input_perceptrons, Layer layer) {
        this.layer = layer;
        this.weights = new HashMap<Perceptron, Float>();
        this.newWeights = new HashMap<Perceptron, Float>();

        Random r = new Random();
        for (Perceptron perceptron : input_perceptrons) {
            Float random = r.nextFloat();
            this.weights.put(perceptron, random);
            System.out.println("\tw: \033[1;93m" + random + "\033[m");
        }

        this.biasWeight = r.nextFloat();
    }

    public void calculateOutput() {
        this.inputSignal = biasWeight;

        for (Perceptron perceptron : weights.keySet()) {
            Float weight = weights.get(perceptron);
            this.inputSignal += perceptron.getOutputSignal() * weight;
        }

        this.outputSignal = this.layer.activate(this.inputSignal);
    }

    public void calculateNewWeights(Float error, Float alpha) {
        this.error = error;

        for (Perceptron perceptron : weights.keySet()) {
            Float weight = alpha * this.error * perceptron.getOutputSignal();
            this.newWeights.put(perceptron, weight);
        }

        this.biasWeight = alpha * error;
    }

    public void setInput(Float value) {
        this.inputSignal = value;
    }

    public void updateWeights() {
        for (Perceptron perceptron :  weights.keySet()) {
            Float previousWeight = weights.get(perceptron);
            Float newWeight = newWeights.get(perceptron);
            this.weights.put(perceptron, previousWeight + newWeight);
        }
    }

    public Float getOutputSignal() {
        return this.outputSignal;
    }

    public Float getInputSignal() {
        return this.inputSignal;
    }

    public Float getError() {
        return this.error;
    }

    public void setOutputSignal(Float signal) {
        this.outputSignal = signal;
    }

    public HashMap<Perceptron, Float> getWeights() {
        return weights;
    }
}