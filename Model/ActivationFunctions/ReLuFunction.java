package Model.ActivationFunctions;

public class ReLuFunction implements ActivatorFunction {
    @Override
    public Float activate(Float signal) {
        if (signal > 0F) {
            return signal;
        }

        return 0F;
    }

    @Override
    public Float derivative(Float signal) {
        if (signal > 0F) {
            return 1F;
        }

        return 0F;
    }

    @Override
    public String getFunctionName() {
        return "ReLu";
    }
}
