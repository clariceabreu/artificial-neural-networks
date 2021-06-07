package IO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Dataset {
    private List<DataVector> vectors;
    private int inputLength;
    private int labelLength;

    public Dataset(String filePath, int labelLength) throws RuntimeException {
        vectors = new ArrayList<>();
        this.labelLength = labelLength;

        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().replaceAll("\\uFEFF", ""); //Remove ZERO WIDTH NO-BREAK SPACE
                String[] entry = line.split(",");
                addVectorFromEntry(entry, labelLength);
            }
        } catch (Exception e) {
            System.out.println("An error occurred while initializing dataset");
            throw new RuntimeException(e);
        }
    }

    private void addVectorFromEntry(String[] entry, int labelLength) {
        this.inputLength = entry.length - this.labelLength;
        float[] input = new float[inputLength];
        float[] label = new float[labelLength];

        for (int i = 0; i < this.inputLength; i++) {
            input[i] = (float) Integer.parseInt(entry[i].trim());
        }

        for (int i = 0; i < this.labelLength; i++) {
            int index = inputLength + i;
            label[i] = (float) Integer.parseInt(entry[index].trim());
        }

        vectors.add(new DataVector(input, label));
    }

    public List<DataVector> getTestSet() {
        return vectors;
    }

    public List<DataVector> getTrainSet() {
        return vectors;
    }

    public int getInputLength() {
        return this.inputLength;
    }

    public int getLabelLength() {
        return this.labelLength;
    }
}
