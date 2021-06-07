package Model.Components;

import IO.DataVector;
import Model.ActivationFunctions.ActivatorFunction;

import java.util.List;
import java.util.ArrayList;

public class Layer
{
    private List<Perceptron> perceptrons;
    private ActivatorFunction function;
    private Layer previousLayer;

    //Pesos random
    public Layer(int numberOfPerceptrons, Layer previousLayer, ActivatorFunction function) {
        this.perceptrons = new ArrayList<>();
        this.function = function;
        this.previousLayer = previousLayer;

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

    public void setOutput(float[] data) {
        for (int i = 0; i < perceptrons.size(); i++) {
            perceptrons.get(i).setOutputSignal(data[i]);
        }
    }

    public void calculateOutput() {
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
            Float outputSignal = perceptrons.get(i).getOutputSignal();
            Float inputSignal = perceptrons.get(i).getInputSignal();
            Float error = (label[i] - outputSignal)  * function.derivative(inputSignal);
            perceptrons.get(i).calculateNewWeights(error, alpha);
        }
    }

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

    public List<Perceptron> getPerceptrons() {
        return this.perceptrons;
    }

    public ActivatorFunction getFunction() {
        return this.function;
    }

    public void setFunction(ActivatorFunction function) {
        this.function = function;
    }

    public void printWeights() {
        int index = 1;
        for (Perceptron perceptron : this.perceptrons) {
            System.out.println("    Input weights for perceptron " + index + ":");
            for (Float weight : perceptron.getWeights().values()) {
                System.out.println("        " + weight);
            }
            System.out.println("        " + perceptron.getBiasWeight() + " (bias)");
            index++;
        }
    }

    //Calculate instant error: E{n} = 1/2 * Σ(target - output)²
    public Float getInstantError(DataVector data) {
        Float errorSum = 0.0F;
        for (int i = 0; i < data.getLabel().length; i++) {
            Float target = data.getLabel()[i];
            Float output = this.perceptrons.get(i).getOutputSignal();
            errorSum += (target - output) * (target - output);
        }

        return 0.5F * errorSum;
    }

    //Calculate mean error: E{av} = 1/N * Σ E{n}
    public Float getMeanSquareError(List<Float> instantErrors) {
        //When there are no instant errors it means that none epochs has passed therefore it returns an error of 1
        if (instantErrors.size() == 0) return 1F;

        Float errorSum = 0.0F;
        for (Float error : instantErrors) {
            errorSum += error;
        }

        return errorSum / instantErrors.size();
    }

    public List<String> getOutput() {
        List<String> outputs = new ArrayList<>();
        for (Perceptron perceptron : this.perceptrons) {
            outputs.add(perceptron.getOutputSignal().toString());
        }

        return outputs;
    }
}
