import java.util.*;

public class Layer
{
    private List<Perceptron> perceptrons;
    public ActivatorFunction function;

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

    public void setData(float[] data) {
        for (int i = 0; i < perceptrons.size(); i++) {
            perceptrons.get(i).setOutputSignal(data[i]);
        }
    }

    public void calculateData() {
        for (Perceptron p : perceptrons) {
            p.calculateOutput();
        }
    }

    public void updateWeights() {
        for (Perceptron p : perceptrons) {
            p.updateWeights();
        }
    }

    public void calculateErrorsFromLabel(float alpha, float[] label) {
        for (int i = 0; i < perceptrons.size(); i++) {
            Perceptron outputPerceptron = perceptrons.get(i);
            Float outputSignal = outputPerceptron.getOutputSignal();
            Float inputSignal = outputPerceptron.getInputSignal();
            Float error = (label[i] - outputSignal)  * function.derived(inputSignal);
            outputPerceptron.calculateNewWeights(error, alpha);
        }
    }

    public void calculateErrors(float alpha, Layer lastLayer) {
        for (Perceptron p : perceptrons) {
            Float errorIn = 0.0F;

            for (Perceptron op : lastLayer.getPerceptrons()) {
                Float weight = op.getWeights().get(p);
                errorIn += weight * op.getError();
            }

            Float error = errorIn * function.derived(p.getInputSignal());

            p.calculateNewWeights(error, alpha);
        }
    }

    public List<Perceptron> getPerceptrons() {
        return perceptrons;
    }
}
