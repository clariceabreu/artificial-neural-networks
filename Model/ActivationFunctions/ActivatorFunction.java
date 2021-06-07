package Model.ActivationFunctions;

public interface ActivatorFunction {
    Float activate(Float signal);
    Float derivative(Float signal);
    String getFunctionName();
}
