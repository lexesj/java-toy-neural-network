package io.github.lexesjan.machinelearning;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Image extends Data {
    private long[] shape;
    boolean isFlat;
    public Image(INDArray[] data) {
        this(data[0], data[1]);
    }

    public Image(INDArray imageData, INDArray label) {
        super(imageData, label);
        this.shape = imageData.shape();
        this.isFlat = false;
    }

    public INDArray getImageData() {
        return super.input;
    }

    public INDArray getLabel() {
        return expectedOutput.argMax();
    }

    public void flatten() {
        super.input = input.reshape(shape[0] * shape[1], 1);
        this.isFlat = true;
    }

    public void unflatten() {
        super.input = input.reshape(shape[0], shape[1]);
        this.isFlat = false;
    }

    public boolean isFlat() {
        return isFlat;
    }

    public long[] shape() {
        return shape;
    }
}
