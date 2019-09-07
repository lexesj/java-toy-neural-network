package io.github.lexesjan.machinelearning;

import io.github.lexesjan.machinelearning.datawrapper.Data;
import io.github.lexesjan.machinelearning.util.Transforms;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.simple.SimpleMatrix;
import java.lang.StringBuilder;
import java.util.Random;

public class NeuralNetwork {
    private SimpleMatrix[] weights;
    private SimpleMatrix[] biases;
    private int[] sizes;
    private int numLayers;

    public NeuralNetwork(int[] sizes) {
        this.weights = new SimpleMatrix[sizes.length - 1];
        this.biases = new SimpleMatrix[sizes.length - 1];
        Random rand = new Random();
        for (int k = 0, j = 1; k < sizes.length - 1; k++, j++) {
            this.weights[k] = SimpleMatrix.wrap(RandomMatrices_DDRM.rectangleGaussian(sizes[j], sizes[k], 0, 1, rand));
            this.biases[k] = SimpleMatrix.wrap(RandomMatrices_DDRM.rectangleGaussian(sizes[j], 1, 0, 1, rand));
        }
        this.sizes = sizes;
        this.numLayers = sizes.length;
    }

    public void train(int epochs, float learningRate, int miniBatchSize, Data[] trainingData) {
        this.train(epochs, learningRate, miniBatchSize, trainingData, null);
    }

    public void train(int epochs, float learningRate, int miniBatchSize, Data[] trainingData, Data[] testData) {
        int n = trainingData.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            long start = System.currentTimeMillis();
            Data[] shuffledTrainingData = shuffle(trainingData);
            Data[][] miniBatches = new Data[(int) Math.ceil((float) n / miniBatchSize)][];
            for (int k = 0, j = 0; k < n; k += miniBatchSize, j++) {
                miniBatches[j] = subArray(shuffledTrainingData, k, k + miniBatchSize);
            }
            for (Data[] miniBatch : miniBatches) {
                updateMiniBatch(miniBatch, learningRate);
            }
            long end = System.currentTimeMillis();
            if (testData == null) {
                System.out.printf("Epoch %d/%d complete. Completed in %d ms\n", epoch + 1, epochs, end - start);
            } else {
                System.out.printf("Epoch %d/%d complete. Accuracy: %.2f. Completed in %d ms\n", epoch + 1, epochs, getAccuracy(testData), end - start);
            }
        }
    }

    private void updateMiniBatch(Data[] miniBatch, float learningRate) {
        SimpleMatrix[] nablaW = new SimpleMatrix[this.weights.length];
        SimpleMatrix[] nablaB = new SimpleMatrix[this.biases.length];
        for (int l = 0; l < this.numLayers - 1; l++) {
            nablaW[l] = new SimpleMatrix(this.weights[l].numRows(), this.weights[l].numCols());
            nablaB[l] = new SimpleMatrix(this.biases[l].numRows(), this.biases[l].numCols());
        }
        for (Data data : miniBatch) {
            SimpleMatrix input = data.getInput();
            SimpleMatrix answer = data.getExpectedOutput();
            SimpleMatrix[][] delta = backprop(input, answer);
            SimpleMatrix[] deltaNablaW = delta[0];
            SimpleMatrix[] deltaNablaB = delta[1];
            for (int i = 0; i < this.numLayers - 1; i++) {
                nablaW[i] = nablaW[i].plus(deltaNablaW[i]);
                nablaB[i] = nablaB[i].plus(deltaNablaB[i]);
            }
        }
        for (int i = 0; i < this.numLayers - 1; i++) {
            this.weights[i] = this.weights[i].minus(nablaW[i].divide(miniBatch.length).scale(learningRate));
            this.biases[i] = this.biases[i].minus(nablaB[i].divide(miniBatch.length).scale(learningRate));
        }
    }

    private SimpleMatrix[][] backprop(SimpleMatrix input, SimpleMatrix answer) {
        SimpleMatrix[] nablaW = new SimpleMatrix[this.weights.length];
        SimpleMatrix[] nablaB = new SimpleMatrix[this.biases.length];
        SimpleMatrix[] zs = new SimpleMatrix[this.numLayers - 1];
        SimpleMatrix[] activations = new SimpleMatrix[this.numLayers];
        SimpleMatrix activation = input;
        activations[0] = activation;
        for (int l = 0; l < this.numLayers - 1; l++) {
            SimpleMatrix z = this.weights[l].mult(activation).plus(this.biases[l]);
            activation = Transforms.sigmoid(z);
            zs[l] = z;
            activations[l + 1] = activation;
        }
        SimpleMatrix delta = costDerivative(activations[activations.length - 1], answer).elementMult(Transforms.sigmoidPrime(activations[activations.length - 1]));
        nablaW[nablaW.length - 1] = delta.mult(activations[activations.length - 2].transpose());
        nablaB[nablaB.length - 1] = delta;
        for (int l = 2; l < this.numLayers; l++) {
            SimpleMatrix sigmoidPrime = Transforms.sigmoidPrime(zs[zs.length - l]);
            delta = this.weights[this.weights.length - l + 1].transpose().mult(delta).elementMult(sigmoidPrime);
            nablaW[nablaW.length - l] = delta.mult(activations[activations.length - l - 1].transpose());
            nablaB[nablaB.length - l] = delta;
        }
        return new SimpleMatrix[][] {nablaW, nablaB};
    }

    public float getAccuracy(Data[] testData) {
        if (testData != null) {
            int correct = 0;
            for (Data data : testData) {
                boolean isCorrect = true;
                for (int i = 0; i < data.getExpectedOutput().numRows() && isCorrect; i++) {
                    double correctAnswer = data.getExpectedOutput().get(i, 0);
                    double answer = Math.round(feedForward(data.getInput()).get(i, 0));
                    if (answer != correctAnswer) {
                        isCorrect = false;
                    }
                }
                if (isCorrect) {
                    correct++;
                }
            }
            return (float) correct / testData.length;
        }
        return -1;
    }

    private SimpleMatrix costDerivative(SimpleMatrix lastActivation, SimpleMatrix answer) {
        return lastActivation.minus(answer);
    }

    public SimpleMatrix feedForward(SimpleMatrix input) {
        SimpleMatrix answer = input;
        for (int l = 0; l < this.numLayers - 1; l++) {
            answer = Transforms.sigmoid(this.weights[l].mult(answer).plus(this.biases[l]));
        }
        return answer;
    }

    private static Data[] shuffle(Data[] unshuffled) {
        Data[] shuffled = new Data[unshuffled.length];
        System.arraycopy(unshuffled, 0, shuffled, 0, shuffled.length);
        for (int i = 0; i < shuffled.length; i++) {
            int index1 = (int) (Math.random() * shuffled.length);
            int index2 = (int) (Math.random() * shuffled.length);
            Data temp = shuffled[index1];
            shuffled[index1] = shuffled[index2];
            shuffled[index2] = temp;
        }
        return shuffled;
    }

    private static Data[] subArray(Data[] array, int start, int passedEnd) {
        int end = Math.min(array.length, passedEnd);
        int size = end - start;
        Data[] subArray = new Data[size];
        for (int i = start, j = 0; i < end && j < subArray.length && i < array.length; i++, j++) {
            subArray[j] = array[i];
        }
        return subArray;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Weights: \n");

        for (SimpleMatrix weight : weights) {
            stringBuilder.append(weight);
            stringBuilder.append("\n");
        }

        stringBuilder.append("\nBiases: \n");
        for (SimpleMatrix bias : biases) {
            stringBuilder.append(bias);
            stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }
}
