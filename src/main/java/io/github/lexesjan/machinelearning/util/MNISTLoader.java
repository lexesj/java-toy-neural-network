package io.github.lexesjan.machinelearning.util;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;
import io.github.lexesjan.machinelearning.datawrapper.Image;
import org.ejml.simple.SimpleMatrix;

public class MNISTLoader {
    private static final int IMAGE = 2051;
    private static final int TRAINING_SET_SIZE = 60000;

    public static SimpleMatrix[][][] loadData(String folder) throws IOException {
        return loadData(new File(folder));
    }

    public static SimpleMatrix[][][] loadData(File folder) throws IOException {
        SimpleMatrix[][][] data = new SimpleMatrix[2][2][];
        File[] files = folder.listFiles();
        SimpleMatrix[] images;
        SimpleMatrix[] labels;
        for (File file : files) {
            DataInputStream fileStream = new DataInputStream(new GZIPInputStream(new FileInputStream(file)));
            int magicNumber = fileStream.readInt();
            int numItems = fileStream.readInt();
            if (magicNumber == IMAGE) {
                images = getImages(fileStream, numItems);
                if (numItems == TRAINING_SET_SIZE) {
                    data[0][0] = images;
                } else {
                    data[1][0] = images;
                }
            } else {
                labels = getLabels(fileStream, numItems);
                if (numItems == TRAINING_SET_SIZE) {
                    data[0][1] = labels;
                } else {
                    data[1][1] = labels;
                }
            }
        }
        return data;
    }

    public static Image[][] loadDataWrapper(String folder) throws IOException {
        return loadDataWrapper(new File(folder));
    }

    public static Image[][] loadDataWrapper(File folder) throws IOException {
        SimpleMatrix[][][] MNISTData = loadData(folder);
        Image[] trainingData = new Image[MNISTData[0][0].length];
        Image[] testData = new Image[MNISTData[1][0].length];
        for (int i = 0; i < trainingData.length; i++) {
            SimpleMatrix image = MNISTData[0][0][i];
            SimpleMatrix label = MNISTData[0][1][i];
            trainingData[i] = new Image(image, label);
            trainingData[i].flatten();
            trainingData[i].normalise();
        }

        for (int i = 0; i < testData.length; i++) {
            SimpleMatrix image = MNISTData[1][0][i];
            SimpleMatrix label = MNISTData[1][1][i];
            testData[i] = new Image(image, label);
            testData[i].flatten();
            testData[i].normalise();
        }
        return new Image[][] {trainingData, testData};
    }

    private static SimpleMatrix[] getImages(DataInputStream fileStream, int numItems) throws IOException {
        SimpleMatrix[] images = new SimpleMatrix[numItems];
        int numRows = fileStream.readInt();
        int numColumns = fileStream.readInt();
        byte[] buffer = new byte[numRows * numColumns];
        for (int i = 0; i < numItems; i++) {
            fileStream.readFully(buffer);
            images[i] = new SimpleMatrix(numRows, numColumns, true, byteArrayToFloatArray(buffer));
        }
        return images;
    }

    private static SimpleMatrix[] getLabels(DataInputStream fileStream, int numItems) throws IOException {
        SimpleMatrix[] labels = new SimpleMatrix[numItems];
        byte[] buffer = new byte[numItems];
        fileStream.readFully(buffer);
        for (int i = 0; i < buffer.length; i++) {
            labels[i] = vectorise(buffer[i]);
        }
        return labels;
    }

    private static SimpleMatrix vectorise(int label) {
        SimpleMatrix result = new SimpleMatrix(10, 1);
        result.set(label, 0, 1);
        return result;
    }

    private static float[] byteArrayToFloatArray(byte[] array) {
        float[] result = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (int) array[i] & 0xFF;
        }
        return result;
    }
}
