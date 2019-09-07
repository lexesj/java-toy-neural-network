package io.github.lexesjan.machinelearning.examples;

import io.github.lexesjan.machinelearning.NeuralNetwork;
import io.github.lexesjan.machinelearning.datawrapper.Data;
import io.github.lexesjan.machinelearning.datawrapper.Image;
import io.github.lexesjan.machinelearning.util.MNISTLoader;

import java.io.IOException;

public class MNIST {
    public static void main(String[] args) throws IOException {
        try {
            long start = System.currentTimeMillis();
            Data[][] MNISTData = MNISTLoader.loadDataWrapper("mnist dataset");
            System.out.println("Finished loading data in " + (System.currentTimeMillis() - start) + " ms.");
            Data[] trainingData = MNISTData[0];
            Data[] testData = MNISTData[1];
            NeuralNetwork nn = new NeuralNetwork(new int[] {((Image) trainingData[0]).numRows(), 30, 10});
            nn.train(30, 3, 10, trainingData, testData);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
