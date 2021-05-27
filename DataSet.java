import java.util.*;

public class DataSet
{
	List<Vector> vectors;

	DataSet(String filename) {
		//TODO: Load from csv file
		vectors = new ArrayList<Vector>();
		vectors.add(new Vector(new float[]{1.0F, 1.0F}, new float[]{1.0F}));
		vectors.add(new Vector(new float[]{1.0F, 0.0F}, new float[]{0.0F}));
		vectors.add(new Vector(new float[]{0.0F, 1.0F}, new float[]{0.0F}));
		vectors.add(new Vector(new float[]{0.0F, 0.0F}, new float[]{0.0F}));
	}

	List<Vector> getTestSet() {
		return vectors;
	}

	List<Vector> getTrainSet() {
		return vectors;
	}
}

class Vector
{
	float[] input;
	float[] label;

	Vector(float[] input, float[] label) {
		this.input = input;
		this.label = label;
	}
}
