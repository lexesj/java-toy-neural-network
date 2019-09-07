package io.github.lexesjan.machinelearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.io.IOException;
import java.lang.StringBuilder;

public class NeuralNetwork {
    private INDArray[] weights;
    private INDArray[] biases;
    private int[] sizes;
    private int numLayers;

    public NeuralNetwork(int[] sizes) {
        this.weights = new INDArray[sizes.length - 1];
        this.biases = new INDArray[sizes.length - 1];
        for (int k = 0, j = 1; k < sizes.length - 1; k++, j++) {
            this.weights[k] = Nd4j.randn(sizes[j], sizes[k]);
            this.biases[k] = Nd4j.randn(sizes[j], 1);
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
        INDArray[] nablaW = new INDArray[this.weights.length];
        INDArray[] nablaB = new INDArray[this.biases.length];
        for (int l = 0; l < this.numLayers - 1; l++) {
            nablaW[l] = Nd4j.zeros(this.weights[l].shape());
            nablaB[l] = Nd4j.zeros(this.biases[l].shape());
        }
        for (Data data : miniBatch) {
            INDArray input = data.getInput();
            INDArray answer = data.getExpectedOutput();
            INDArray[][] delta = backprop(input, answer);
            INDArray[] deltaNablaW = delta[0];
            INDArray[] deltaNablaB = delta[1];
            for (int i = 0; i < this.numLayers - 1; i++) {
                nablaW[i].addi(deltaNablaW[i]);
                nablaB[i].addi(deltaNablaB[i]);
            }
        }
        for (int i = 0; i < this.numLayers - 1; i++) {
            this.weights[i].subi(nablaW[i].divi(miniBatch.length).muli(learningRate));
            this.biases[i].subi(nablaB[i].divi(miniBatch.length).muli(learningRate));
        }
    }

    private INDArray[][] backprop(INDArray input, INDArray answer) {
        INDArray[] nablaW = new INDArray[this.weights.length];
        INDArray[] nablaB = new INDArray[this.biases.length];
        INDArray[] zs = new INDArray[this.numLayers - 1];
        INDArray[] activations = new INDArray[this.numLayers];
        INDArray activation = input;
        activations[0] = activation;
        for (int l = 0; l < this.numLayers - 1; l++) {
            INDArray z = this.weights[l].mmul(activation).add(this.biases[l]);
            activation = Transforms.sigmoid(z);
            zs[l] = z;
            activations[l + 1] = activation;
        }
        INDArray delta = costDerivative(activations[activations.length - 1], answer).mul(Transforms.sigmoidDerivative(activations[activations.length - 1]));
        nablaW[nablaW.length - 1] = delta.mmul(activations[activations.length - 2].transpose());
        nablaB[nablaB.length - 1] = delta;
        for (int l = 2; l < this.numLayers; l++) {
            INDArray sigmoidPrime = Transforms.sigmoidDerivative(zs[zs.length - l]);
            delta = this.weights[this.weights.length - l + 1].transpose().mmul(delta).mul(sigmoidPrime);
            nablaW[nablaW.length - l] = delta.mmul(activations[activations.length - l - 1].transpose());
            nablaB[nablaB.length - l] = delta;
        }
        return new INDArray[][] {nablaW, nablaB};
    }

    private float getAccuracy(Data[] testData) {
        int correct = 0;
        for (Data data : testData) {
            long[] shape = data.getExpectedOutput().shape();
            boolean isCorrect = true;
            for (int i = 0; i < shape[0] && isCorrect; i++) {
                float correctAnswer = data.getExpectedOutput().getFloat(i, 1);
                float answer = Math.round(feedForward(data.getInput()).getFloat(i, 1));
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

    private INDArray costDerivative(INDArray lastActivation, INDArray answer) {
        return lastActivation.sub(answer);
    }

    private INDArray feedForward(INDArray input) {
        INDArray answer = input;
        for (int l = 0; l < this.numLayers - 1; l++) {
            answer = Transforms.sigmoid(this.weights[l].mmul(answer).add(this.biases[l]));
        }
        return answer;
    }

    private static float sigmoid(float z) {
        return 1 / (1 + (float) Math.exp(-z));
    }

    private static float sigmoidPrime(float z) {
        return sigmoid(z) * (1 - sigmoid(z));
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

        for (INDArray weight : weights) {
            stringBuilder.append(weight);
            stringBuilder.append("\n");
        }

        stringBuilder.append("\nBiases: \n");
        for (INDArray bias : biases) {
            stringBuilder.append(bias);
            stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }

    public static void main(String[] args) {
        Data[] trainingData = {
                new Data(Nd4j.create(new float[] {0, 0}, new int[] {2, 1}), Nd4j.create(new float[] {0}, new int[] {1, 1})),
                new Data(Nd4j.create(new float[] {0, 1}, new int[] {2, 1}), Nd4j.create(new float[] {1}, new int[] {1, 1})),
                new Data(Nd4j.create(new float[] {1, 0}, new int[] {2, 1}), Nd4j.create(new float[] {1}, new int[] {1, 1})),
                new Data(Nd4j.create(new float[] {1, 1}, new int[] {2, 1}), Nd4j.create(new float[] {0}, new int[] {1, 1}))
        };
        NeuralNetwork nn = new NeuralNetwork(new int[] {2, 10, 1});
        nn.train(10000, 3, 1, trainingData, trainingData);
        System.out.println(nn.feedForward(Nd4j.create(new float[] {0, 0}, new int[] {2, 1})));
        System.out.println(nn.feedForward(Nd4j.create(new float[] {0, 1}, new int[] {2, 1})));
        System.out.println(nn.feedForward(Nd4j.create(new float[] {1, 0}, new int[] {2, 1})));
        System.out.println(nn.feedForward(Nd4j.create(new float[] {1, 1}, new int[] {2, 1})));
//        try {
//            long start = System.currentTimeMillis();
//            Data[][] MNISTData = MNISTLoader.loadDataWrapper("mnist dataset");
//            System.out.println("Finished loading data in " + (System.currentTimeMillis() - start) + " ms.");
//            Data[] trainingData = MNISTData[0];
//            Data[] testData = MNISTData[1];
//            long[] shape = ((Image) trainingData[0]).shape();
//            NeuralNetwork nn = new NeuralNetwork(new int[] {(int) (shape[0] * shape[1]), 16, 10});
//            nn.train(30, 3, 128, trainingData, testData);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }
}
