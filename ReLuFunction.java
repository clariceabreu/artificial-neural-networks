public class ReLuFunction implements ActivatorFunction {
    @Override public Float activate(Float signal) {
        if (signal > 0F) {
            return signal;
        }

        return 0F;
    }

    @Override public Float derived(Float signal) {
        if (signal > 0F) {
            return 1F;
        }

        return 0F;
    }
}
