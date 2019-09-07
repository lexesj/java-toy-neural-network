package io.github.lexesjan.machinelearning.util;

import org.ejml.simple.SimpleMatrix;

public class Transforms {
    public static SimpleMatrix sigmoid(SimpleMatrix z){
        int numRows = z.numRows();
        int numCols = z.numCols();
        SimpleMatrix result = new SimpleMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                result.set(i, j, sigmoid(z.get(i, j)));
            }
        }
        return result;
    }

    public static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public static SimpleMatrix sigmoidPrime(SimpleMatrix z){
        int numRows = z.numRows();
        int numCols = z.numCols();
        SimpleMatrix result = new SimpleMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                result.set(i, j, sigmoidPrime(z.get(i, j)));
            }
        }
        return result;
    }

    public static double sigmoidPrime(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
}
