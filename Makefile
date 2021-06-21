SOURCES=IO/DataVector.java IO/Dataset.java IO/Output.java Main.java Model/ActivationFunctions/ActivatorFunction.java Model/ActivationFunctions/ReLuFunction.java Model/ActivationFunctions/SigmoidFunction.java Model/Components/Layer.java Model/Components/Perceptron.java Model/Model.java

.PHONY: all run plot clean

all: Main.class

Main.class: $(SOURCES)
	javac $^

run: Main.class
	java Main datasets/dataset_XOR.csv 1

run-charset: Main.class
	java Main datasets/dataset_chars_clean.csv datasets/dataset_chars_clean.csv 7

run-charset-noise: Main.class
	java Main datasets/dataset_chars_clean.csv datasets/dataset_chars_noise.csv 7

run-charset-noise-20: Main.class
	java Main datasets/dataset_chars_clean.csv datasets/dataset_chars_noise_20.csv 7

plot:
	gnuplot --persist -e 'plot "outputs/train_errors.txt" with lines, "outputs/test_errors.txt" with lines'


clean:
	find -name *.class -exec rm {} \;
