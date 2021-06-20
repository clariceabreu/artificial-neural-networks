package IO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Scanner;

public class Dataset {
    private List<DataVector> vectorsTrain;
    private List<DataVector> vectorsTest;
    private int inputLength;
    private int labelLength;

    public Dataset(String trainDataset, String testDataset, int labelLength) throws RuntimeException {
        vectorsTrain = new ArrayList<>();
        vectorsTest = new ArrayList<>();
        this.labelLength = labelLength;

        this.getEntryFromFile(trainDataset, vectorsTrain);
        this.getEntryFromFile(testDataset, vectorsTest);
    }

    private void getEntryFromFile(String filePath, List<DataVector> vectors) {
        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().replaceAll("\\uFEFF", ""); //Remove ZERO WIDTH NO-BREAK SPACE
                String[] entry = line.split(",");
                addVectorFromEntry(entry, labelLength, vectors);
            }
        } catch (Exception e) {
            System.out.println("An error occurred while initializing dataset");
            throw new RuntimeException(e);
        }
    }

    private void addVectorFromEntry(String[] entry, int labelLength, List<DataVector> vectors) {
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
        return vectorsTest;
    }

    public List<DataVector> getTrainSet() {
        Collections.shuffle(vectorsTrain);
        return vectorsTrain;
    }

    public int getInputLength() {
        return this.inputLength;
    }

    public int getLabelLength() {
        return this.labelLength;
    }
}
