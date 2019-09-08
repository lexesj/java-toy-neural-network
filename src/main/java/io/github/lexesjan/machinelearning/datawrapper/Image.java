package io.github.lexesjan.machinelearning.datawrapper;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.ejml.simple.SimpleMatrix;

public class Image extends Data {
    private boolean isFlat;
    private int[] originalShape;

    public Image(double[][] imageData, double[][] label) {
        super(new SimpleMatrix(imageData), new SimpleMatrix(label));
    }

    public Image(SimpleMatrix[] data) {
        this(data[0], data[1]);
    }

    public Image(SimpleMatrix imageData, SimpleMatrix label) {
        super(imageData, label);
        this.originalShape = new int[]{imageData.numRows(), imageData.numCols()};
        this.isFlat = false;
    }

    public SimpleMatrix getImageData() {
        return super.input;
    }

    public int getLabel() {
        double max = -1;
        int maxIndex = -1;
        for (int i = 0; i < super.expectedOutput.numRows(); i++) {
            double currentOutput = super.expectedOutput.get(i, 0);
            if (Math.max(max, currentOutput) != max) {
                maxIndex = i;
            }
            max = Math.max(currentOutput, max);
        }
        return maxIndex;
    }

    public void flatten() {
        super.input.reshape((originalShape[0] * originalShape[1]), 1);
        this.isFlat = true;
    }

    public void unflatten() {
        super.input.reshape(originalShape[0], originalShape[1]);
        this.isFlat = false;
    }

    public boolean isFlat() {
        return isFlat;
    }

    public boolean isNormalised() {
        return isNormalised;
    }

    public void save(String fileName, String fileExtension) throws IOException {
        save(new File(fileName), fileExtension);
    }

    public void save(File file, String fileExtension) throws IOException {
        if (this.isFlat())
            this.unflatten();
        if (this.isNormalised())
            this.unnormalise();
        int rows = numRows();
        int columns = numCols();
        BufferedImage img = new BufferedImage(rows, columns, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < columns; y++) {
                img.setRGB(y, x, (byte) super.input.get(x, y));
            }
        }
        ImageIO.write(img, fileExtension, file);
    }

    public int numRows() {
        return super.input.numRows();
    }

    public int numCols() {
        return super.input.numCols();
    }
}
