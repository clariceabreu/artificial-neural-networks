package IO;

public class DataVector {
    private float[] input;
    private float[] label;

    public DataVector(float[] input, float[] label) {
        this.input = input;
        this.label = label;
    }

    public float[] getInput() {
        return input;
    }

    public float[] getLabel() {
        return label;
    }
}
