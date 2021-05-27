SRC=ActivatorFunction.java Layer.java Model.java Perceptron.java SigmoidFunction.java
OUTPUTS=ActivatorFunction.class Layer.class Model.class Perceptron.class SigmoidFunction.class

.PHONY: all run clean

all: $(OUTPUTS)

%.class: %.java
	javac $<

run:
	java Model 2 2 1

clean:
	rm -f *.class
