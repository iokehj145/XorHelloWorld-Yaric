import java.util.Arrays;
import java.util.Random;
public class NeuralNetwork {
    private Neuron[] hiddenLayer;
    private int hiddenLen;
    private double[][] weights1;
    private double[] biases1;
    private double[] weights2;
    private double bias2;
    public NeuralNetwork(int inputSize, int hiddenLayerSize) {
        Random random = new Random();
        hiddenLen = hiddenLayerSize;
        hiddenLayer = new Neuron[hiddenLen];
        weights1 = new double[2][hiddenLen];
        biases1 = new double[hiddenLen];
        weights2 = new double[hiddenLen];
        bias2 = 0.0;
        for (int i = 0; i < weights1.length; i++) {
            for (int j = 0; j < weights1[i].length; j++) {
                weights1[i][j] = random.nextDouble() * 2 - 1;
            }
        }

        for (int i = 0; i < weights2.length; i++) {
            weights2[i] = random.nextDouble() * 2 - 1;
        }

        for (int i = 0; i < hiddenLayerSize; i++) {
            hiddenLayer[i] = new Neuron(inputSize);
        }
    }
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public void train(double[][] inputs, double[] targets, double learningRate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Прямой проход
                double[] hidden = new double[hiddenLen];
                for (int j = 0; j < hiddenLen; j++) {
                    double sum = biases1[j];
                    for (int k = 0; k < 2; k++) {
                        sum += inputs[i][k] * weights1[k][j];
                    }
                    hidden[j] = hiddenLayer[j].sigmoid(sum);
                }

                double output = bias2;
                for (int j = 0; j < hiddenLen; j++) {
                    output += hidden[j] * weights2[j];
                }
                output = sigmoid(output);

                // Обратный проход
                double error = targets[i] - output;
                bias2 += learningRate * error * sigmoidDerivative(output);
                for (int j = 0; j < hiddenLen; j++) {
                    weights2[j] += learningRate * error * sigmoidDerivative(output) * hidden[j];
                    double delta = learningRate * error * sigmoidDerivative(output) * weights2[j];
                    for (int k = 0; k < 2; k++) {
                        weights1[k][j] += delta * sigmoidDerivative(hidden[j]) * inputs[i][k];
                        biases1[j] += delta * sigmoidDerivative(hidden[j]);
                    }
                }
            }
        }
}


    public double feedforward(double[] inputs) {
        double[] hidden = new double[hiddenLen];
        for (int j = 0; j < hiddenLen; j++) {
            double sum = biases1[j];
            for (int k = 0; k < inputs.length; k++) {
                sum += inputs[k] * weights1[k][j];
            }
            hidden[j] = sigmoid(sum);
        }

        double output = bias2;
        for (int j = 0; j < hiddenLen; j++) {
            output += hidden[j] * weights2[j];
        }
        return sigmoid(output);
    }
}