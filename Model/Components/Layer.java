package Model.Components;

import IO.DataVector;
import Model.ActivationFunctions.ActivatorFunction;

import java.util.List;
import java.util.ArrayList;

public class Layer {
    private List<Perceptron> perceptrons;
    private ActivatorFunction function;
    private Float meanSquareError;

    //Instantiates all the perceptrons of the layer using random weights
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

    //Instantiates all the perceptrons of the layer using fixed weights
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

    //Sets outputs of the perceptrons using the data from the dataset (this method is only used for input layer)
    public void setOutput(float[] data) {
        for (int i = 0; i < perceptrons.size(); i++) {
            perceptrons.get(i).setOutputSignal(data[i]);
        }
    }

    //Calls calculateOutput() from all perceptrons dataIndexof the layer
    public void calculateOutput() { this.perceptrons.forEach(Perceptron::calculateOutput); }

    //Calls updateWeights() from all perceptrons of the layer
    public void updateWeights() { this.perceptrons.forEach(Perceptron::updateWeights); }

    //Calculates the error using the labels in the dataset.
    //δ{k} = (target{k} - output{k}) * f'(input{k})
    //Where k = 1 .. m  / m = number of perceptrons in the output layer
    public void calculateErrorsFromLabel(float alpha, float[] label) {
        for (int i = 0; i < perceptrons.size(); i++) {
            Float outputSignal = perceptrons.get(i).getOutputSignal();
            Float inputSignal = perceptrons.get(i).getInputSignal();
            Float error = (label[i] - outputSignal)  * function.derivative(inputSignal);
            perceptrons.get(i).calculateNewWeights(error, alpha);
        }
    }

    //Calculates the error using the errors from the upper layer.
    //δ_in{j} = Σ δ{k} * weight {j, k}
    //δ{j} = δ_in{j} * f'(input{j})
    //Where k = 1 .. m  / m = number of perceptrons in the upper layer
    //Where j = 1 .. p / p = number of perceptrons in the current layer
    public void propagateError(float alpha, List<Perceptron> outputPerceptrons) {
        for (Perceptron p : perceptrons) {
            Float errorIn = 0.0F;

            for (Perceptron op : outputPerceptrons) {
                Float weight = op.getWeights().get(p);
                errorIn += weight * op.getError();
            }

            Float error = errorIn * function.derivative(p.getInputSignal());

            p.calculateNewWeights(error, alpha);
        }
    }

    //Calculate instant error
    //E{n} = 1/2 * Σ(target - output)²
    public Float calculateInstantError(DataVector data) {
        Float errorSum = 0.0F;
        for (int i = 0; i < data.getLabel().length; i++) {
            Float target = data.getLabel()[i];
            Float output = this.perceptrons.get(i).getOutputSignal();
            errorSum += (target - output) * (target - output);
        }

        return  0.5F * errorSum;
    }

    //Calculate mean error: E{av} = 1/N * Σ E{n}
    //Where n = 1 .. N / N = number of entries in the dataset
    public Float calculateMeanSquareError(List<Float> instantErrors) {
        Float errorSum = 0.0F;
        for (Float error : instantErrors) {
            errorSum += error;
        }

        this.meanSquareError = errorSum / instantErrors.size();
        return this.meanSquareError;
    }

    //Returns an array of the output signals of all the perceptrons
    public String[] getRawOutput() {
        String[] output = new String[this.perceptrons.size()];

        for (int i = 0; i < this.perceptrons.size(); i++) {
            output[i] = perceptrons.get(i).getOutputSignal().toString();
        }

        return output;
    }


    //Returns an array of the output signals of all the perceptrons (converted to int)
    public String[] getOutput() {
        String[] output = new String[this.perceptrons.size()];

        for (int i = 0; i < this.perceptrons.size(); i++) {
            if (this.perceptrons.get(i).getOutputSignal() > 0.9F) {
                output[i] = "1";
            } else {
                output[i] = "0";
            }
        }

        return output;
    }

    public List<Perceptron> getPerceptrons() { return this.perceptrons; }

    public ActivatorFunction getFunction() { return this.function; }

    public Float getMeanSquareError() { return this.meanSquareError; }
}
