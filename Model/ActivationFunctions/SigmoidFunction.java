package Model.ActivationFunctions;

public class SigmoidFunction implements ActivatorFunction {
    public SigmoidFunction() { }

    @Override
    public Float activate(Float signal) {
        return 1.0F / (1.0F + (float) Math.pow(Math.E, -signal));
    }

    @Override
    public Float derivative(Float signal) {
        return activate(signal) * (1.0F - activate(signal));
    }

    @Override
    public String getFunctionName() {
        return "Sigmoid";
    }
}
