package io.github.lexesjan.machinelearning.datawrapper;

import org.ejml.simple.SimpleMatrix;

public class Data {
    protected SimpleMatrix input;
    protected SimpleMatrix expectedOutput;
    protected double max;
    protected boolean isNormalised;

    public Data(SimpleMatrix[] data) {
        this(data[0], data[1]);
    }

    public Data(SimpleMatrix input, SimpleMatrix expectedOutput) {
        this.input = input;
        this.expectedOutput = expectedOutput;
        this.max = this.input.elementMaxAbs();
        this.isNormalised = false;
    }

    public SimpleMatrix getInput() {
        return input;
    }

    public SimpleMatrix getExpectedOutput() {
        return expectedOutput;
    }

    public void normalise() {
        this.isNormalised = true;
        this.input = this.input.divide(max);
    }

    public void unnormalise() {
        this.isNormalised = false;
        for (int i = 0; i < this.input.numRows(); i++) {
            for (int j = 0; j < this.input.numCols(); j++) {
                this.input.set(i, j, this.input.get(i, j) * max);
            }
        }
    }

    @Override
    public String toString() {
        return this.input.toString();
    }
}
