SRC=ActivatorFunction.java DataSet.java Layer.java Model.java Perceptron.java SigmoidFunction.java Main.java
OUTPUTS=ActivatorFunction.class DataSet.class Layer.class Model.class Perceptron.class SigmoidFunction.class Main.class

.PHONY: all run clean

all: $(OUTPUTS)

%.class: %.java
	javac $<

run:
	java Main Datasets/dataset_AND.csv 1

clean:
	rm -f *.class
