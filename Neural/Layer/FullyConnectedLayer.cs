using Skotz.Neural.Activation;
using Skotz.Neural.Utility;
using System;

namespace Skotz.Neural.Layer
{
    public class FullyConnectedLayer : ILayer
    {
        private GaussianRandom _random;
        private IActivation _activationFunction;
        private int _previousLayerSize;
        private int _numberOfNeurons;

        private double[,] _weights;
        private double[] _biases;
        private double[] _inputs;
        private double[] _activations;

        public int Size => _numberOfNeurons;

        public FullyConnectedLayer(int previousLayerSize, int numberOfNeurons, IActivation activation)
        {
            _activationFunction = activation;
            _previousLayerSize = previousLayerSize;
            _numberOfNeurons = numberOfNeurons;

            var stdDev = _activationFunction.StandardDeviation(previousLayerSize, numberOfNeurons);
            _random = new GaussianRandom(0, stdDev);

            _weights = new double[numberOfNeurons, previousLayerSize];
            _biases = new double[numberOfNeurons];

            for (int n = 0; n < numberOfNeurons; n++)
            {
                for (int p = 0; p < previousLayerSize; p++)
                {
                    _weights[n, p] = _random.NextDouble();
                }
            }
        }

        public double[] FeedForward(double[] values)
        {
            // Save the inputs for backprop
            _inputs = values;
            _activations = new double[_numberOfNeurons];

            for (int n = 0; n < _numberOfNeurons; n++)
            {
                for (int p = 0; p < _previousLayerSize; p++)
                {
                    _activations[n] += _inputs[p] * _weights[n, p];
                }

                _activations[n] = _activationFunction.Run(_activations[n] + _biases[n]);
            }

            return _activations;
        }

        public double[] Backpropagate(double[] gradients, double learningRate)
        {
            var nextGradients = new double[_previousLayerSize];

            // Gradients coming in are with respect to the result of the activation function
            for (int n = 0; n < _numberOfNeurons; n++)
            {
                gradients[n] *= _activationFunction.Derivative(_activations[n]);

                // Clip the gradient to avoid diverging to infinity
                gradients[n] = ClipGradient(gradients[n]);
            }

            // Gradients with respect to each weight
            for (int n = 0; n < _numberOfNeurons; n++)
            {
                for (int p = 0; p < _previousLayerSize; p++)
                {
                    var weightGradientNP = gradients[n] * _inputs[p];

                    // Update weights based on gradients
                    _weights[n, p] -= weightGradientNP * learningRate;

                    // Store the correct gradients to pass back to the previous layer
                    nextGradients[p] += weightGradientNP;
                }
            }

            // Gradients for the biases of each neuron
            for (int n = 0; n < _numberOfNeurons; n++)
            {
                // Update biases based on gradients
                _biases[n] -= gradients[n] * learningRate;
            }

            return nextGradients;
        }

        private double ClipGradient(double gradient)
        {
            return Math.Min(Math.Max(gradient, -1), 1);
        }
    }
}