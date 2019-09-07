package io.github.lexesjan.machinelearning;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Data {
    protected INDArray input;
    protected INDArray expectedOutput;

    public Data(INDArray[] data) {
        this(data[0], data[1]);
    }

    public Data(INDArray input, INDArray expectedOutput) {
        this.input = input;
        this.expectedOutput = expectedOutput;
    }

    public INDArray getInput() {
        return input;
    }

    public INDArray getExpectedOutput() {
        return expectedOutput;
    }

    public void normalise() {
        this.input = this.input.div(this.input.maxNumber());
    }
}
