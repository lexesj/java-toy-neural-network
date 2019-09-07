package io.github.lexesjan.machinelearning;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import javax.imageio.ImageIO;

public class MNISTLoader {
    private static final int IMAGE = 2051;
    private static final int TRAINING_SET_SIZE = 60000;

    public static INDArray[][][] loadData(String folder) throws IOException {
        return loadData(new File(folder));
    }

    public static INDArray[][][] loadData(File folder) throws IOException {
        INDArray[][][] data = new INDArray[2][2][];
        File[] files = folder.listFiles();
        INDArray[] images;
        INDArray[] labels;
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
        INDArray[][][] MNISTData = loadData(folder);
        Image[] trainingData = new Image[MNISTData[0][0].length];
        Image[] testData = new Image[MNISTData[1][0].length];
        for (int i = 0; i < trainingData.length; i++) {
            INDArray image = MNISTData[0][0][i];
            INDArray label = MNISTData[0][1][i];
            trainingData[i] = new Image(image, label);
            trainingData[i].flatten();
            trainingData[i].normalise();
        }

        for (int i = 0; i < testData.length; i++) {
            INDArray image = MNISTData[1][0][i];
            INDArray label = MNISTData[1][1][i];
            testData[i] = new Image(image, label);
            testData[i].flatten();
            testData[i].normalise();
        }
        return new Image[][] {trainingData, testData};
    }

    private static INDArray[] getImages(DataInputStream fileStream, int numItems) throws IOException {
        INDArray[] images = new INDArray[numItems];
        int numRows = fileStream.readInt();
        int numColumns = fileStream.readInt();
        byte[] buffer = new byte[numRows * numColumns];
        for (int i = 0; i < numItems; i++) {
            fileStream.readFully(buffer);
            images[i] = Nd4j.create(byteArrayToFloatArray(buffer), new int[] {numRows, numColumns});
        }
        return images;
    }

    private static INDArray[] getLabels(DataInputStream fileStream, int numItems) throws IOException {
        INDArray[] labels = new INDArray[numItems];
        byte[] buffer = new byte[numItems];
        fileStream.readFully(buffer);
        for (int i = 0; i < buffer.length; i++) {
            labels[i] = vectorise(buffer[i]);
        }
        return labels;
    }

    private static INDArray vectorise(int label) {
        INDArray result = Nd4j.zeros(10, 1);
        result.putScalar(new int[] {label, 0}, 1);
        return result;
    }

    private static float[] byteArrayToFloatArray(byte[] array) {
        float[] result = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (int) array[i] & 0xFF;
        }
        return result;
    }

    private static void createImageFromINDArray(INDArray imageArray, File file) throws IOException{
        long[] shape = imageArray.shape();
        int rows = (int) shape[0];
        int columns = (int) shape[1];
        BufferedImage img = new BufferedImage(rows, columns, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < columns; y++) {
                img.setRGB(y, x, (int) imageArray.getFloat(x, y));
            }
        }
        ImageIO.write(img, "png", file);
    }

    public static void main(String[] args) {
        INDArray result = Nd4j.zeros(10, 1);
        result.putScalar(new int[] {2, 0}, 1);
        System.out.println(result);
    }
}
