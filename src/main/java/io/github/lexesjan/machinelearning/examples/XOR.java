package io.github.lexesjan.machinelearning.examples;

import io.github.lexesjan.machinelearning.NeuralNetwork;
import io.github.lexesjan.machinelearning.datawrapper.Data;
import org.ejml.simple.SimpleMatrix;

public class XOR {
    public static void main(String[] args) {
        Data[] trainingData = {
                new Data(new SimpleMatrix(new float[][] {{0}, {0}}), new SimpleMatrix(new float[][] {{0}})),
                new Data(new SimpleMatrix(new float[][] {{0}, {1}}), new SimpleMatrix(new float[][] {{1}})),
                new Data(new SimpleMatrix(new float[][] {{1}, {0}}), new SimpleMatrix(new float[][] {{1}})),
                new Data(new SimpleMatrix(new float[][] {{1}, {1}}), new SimpleMatrix(new float[][] {{0}})),
        };
        NeuralNetwork nn = new NeuralNetwork(new int[] {2, 10, 1});
        nn.train(10000, 3, 1, trainingData, trainingData);
        System.out.println(nn.feedForward(new SimpleMatrix(new float[][] {{0}, {0}})));
        System.out.println(nn.feedForward(new SimpleMatrix(new float[][] {{0}, {1}})));
        System.out.println(nn.feedForward(new SimpleMatrix(new float[][] {{1}, {0}})));
        System.out.println(nn.feedForward(new SimpleMatrix(new float[][] {{1}, {1}})));
    }
}
