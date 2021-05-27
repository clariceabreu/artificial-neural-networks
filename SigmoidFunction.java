public class SigmoidFunction implements ActivatorFunction
{
    public SigmoidFunction() { }

    @Override
    public Float activate(Float signal) {
        return 1.0F / (1.0F + (float) Math.pow(Math.E, -signal));
    }

    @Override
    public Float derived(Float signal) {
        return activate(signal) * (1 - activate(signal));
    }
}
