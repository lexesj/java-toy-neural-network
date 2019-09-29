# java-toy-neural-network
I built this library as I was facinated in neural networks and wondered how they worked internally. This is a simple feed forward neural network which uses the Efficient Java Matrix Library (EJML) to do the matrix caculations.

## Getting started
  ### Prerequisites
  You will need to have installed:
  1. Java
  2. Maven
  
  
  ### Building
  Build the jar file using Maven and the POM xml file using the following code
  ```
  mvn package
  ```
  ### Running Examples
  - MNIST dataset
    1. Download all files of the [MNIST dataset](yann.lecun.com/exdb/mnist/)
    2. Add the library to the classpath
    3. Change the folder location parameter in the loadDataWrapper() method in the example code
    4. Run the MNIST example code
  - XOR problem
    1. Add the library to the classpath
    2. Run the XOR example code
    
  ### Documentation
  - ```NeuralNetwork``` - The neural network class
      - ```NeuralNetwork(int[] sizes)``` - Creates a new neural network with the specified layer sizes (each element in the array corresponds to a layer size)
      - ```SimpleMatrix feedForward(SimpleMatrix input)``` - Returns the output of the neural network as a SimpleMatrix which is a vecto.
      - ```double[][] feedForward(double[][] input)``` - Returns the output of the neural network as a 2d double array which is a vector
      - ```void train(int epochs, float learningRate, int miniBatchSize, Data[] trainingData)``` - Trains the neural network
      - ```void train(int epochs, float learningRate, int miniBatchSize, Data[] trainingData, Data[] testData)``` - Trains the neural network and prints out the accuracy after each epoch
      - ```SimpleMatrix fileToMatrix(String fileName)``` - Takes in an image name file and converts it to a SimpleMatrix object
      - ```SimpleMatrix fileToMatrix(File file)``` - Takes in an image file and converts it to a SimpleMatrix object
      - ```float getAccuracy(Data[] testData)``` - Calculates the accuracy of the neural network with the provided test data
  
  ## Library Used
  [Efficient Java Matrix Library (EJML)](https://github.com/lessthanoptimal/ejml)
  
