using Skotz.Neural.Activation;
using Skotz.Neural.Utility;
using System;

namespace Skotz.Neural.Layer
{
    public class ConvolutionLayer : ILayer
    {
        private GaussianRandom _random;
        private IActivation _activationFunction;

        private int _previousLayerWidth;
        private int _previousLayerHeight;
        private int _previousLayerDepth;

        private int _outputWidth;
        private int _outputHeight;

        private int _featureSize;
        private int _numberOfFeatures;
        private bool _pad;
        private int _stride;

        private double[,,,] _kernels;
        private double[] _biases;
        private double[,,] _inputs;
        private double[,,] _activations;

        public ConvolutionLayer(int previousWidth, int previousHeight, int previousDepth, int featureSize, int numberOfFeatures, IActivation activation)
        {
            _activationFunction = activation;

            _previousLayerWidth = previousWidth;
            _previousLayerHeight = previousHeight;
            _previousLayerDepth = previousDepth;

            _featureSize = featureSize;
            _numberOfFeatures = numberOfFeatures;

            // TODO
            _pad = false;
            _stride = 1;

            _outputWidth = _pad ? _previousLayerWidth : (_previousLayerWidth - _featureSize + 1);
            _outputHeight = _pad ? _previousLayerHeight : (_previousLayerHeight - _featureSize + 1);

            var stdDev = _activationFunction.StandardDeviation(previousWidth, numberOfFeatures);
            _random = new GaussianRandom(0, stdDev);

            _kernels = new double[_numberOfFeatures, _featureSize, _featureSize, _previousLayerDepth];
            _biases = new double[_numberOfFeatures];

            for (int f = 0; f < _numberOfFeatures; f++)
            {
                for (int w = 0; w < _featureSize; w++)
                {
                    for (int h = 0; h < _featureSize; h++)
                    {
                        for (int d = 0; d < _previousLayerDepth; d++)
                        {
                            _kernels[f, w, h, d] = _random.NextDouble();
                        }
                    }
                }
            }
        }

        public double[,,] FeedForward(double[,,] values)
        {
            var kernelBorder = (_featureSize - 1) / 2;
            var padding = _pad ? kernelBorder : 0;

            // Save the inputs for backprop
            _inputs = values;
            _activations = new double[_outputWidth, _outputHeight, _numberOfFeatures];

            for (int f = 0; f < _numberOfFeatures; f++)
            {
                // Input dimensions (centered on the kernel)
                for (int w = kernelBorder - padding; w < _previousLayerWidth - (kernelBorder - padding); w += _stride)
                {
                    for (int h = kernelBorder - padding; h < _previousLayerHeight - (kernelBorder - padding); h += _stride)
                    {
                        // Kernel dimensions
                        for (int kw = 0; kw < _featureSize; kw++)
                        {
                            for (int kh = 0; kh < _featureSize; kh++)
                            {
                                for (int kd = 0; kd < _previousLayerDepth; kd++)
                                {
                                    _activations[w - (kernelBorder - padding), h - (kernelBorder - padding), f] += _kernels[f, kw, kh, kd] * values[w + (kw - kernelBorder), h + (kh - kernelBorder), kd];
                                }
                            }
                        }
                    }
                }
            }

            for (int f = 0; f < _numberOfFeatures; f++)
            {
                for (int w = 0; w < _outputWidth; w += _stride)
                {
                    for (int h = 0; h < _outputHeight; h += _stride)
                    {
                        _activations[w, h, f] = _activationFunction.Run(_activations[w, h, f] + _biases[f]);
                    }
                }
            }

            return _activations;
        }

        public double[,,] Backpropagate(double[,,] gradients, double learningRate)
        {
            var nextGradients = new double[_previousLayerWidth, _previousLayerHeight, _previousLayerDepth];

            var kernelBorder = (_featureSize - 1) / 2;
            var padding = _pad ? kernelBorder : 0;

            // Gradients coming in are with respect to the result of the activation function
            for (int w = 0; w < _outputWidth; w += _stride)
            {
                for (int h = 0; h < _outputHeight; h += _stride)
                {
                    for (int f = 0; f < _numberOfFeatures; f++)
                    {
                        gradients[w, h, f] *= _activationFunction.Derivative(_activations[w, h, f]);

                        // Clip the gradient to avoid diverging to infinity
                        gradients[w, h, f] = ClipGradient(gradients[w, h, f]);
                    }
                }
            }

            // Gradients with respect to each weight in the kernel
            for (int f = 0; f < _numberOfFeatures; f++)
            {
                for (int w = kernelBorder - padding; w < _previousLayerWidth - (kernelBorder - padding); w += _stride)
                {
                    for (int h = kernelBorder - padding; h < _previousLayerHeight - (kernelBorder - padding); h += _stride)
                    {
                        for (int kw = 0; kw < _featureSize; kw++)
                        {
                            for (int kh = 0; kh < _featureSize; kh++)
                            {
                                for (int kd = 0; kd < _previousLayerDepth; kd++)
                                {
                                    var weightGradientWHF = gradients[w - (kernelBorder - padding), h - (kernelBorder - padding), f] * _inputs[w + (kw - kernelBorder), h + (kh - kernelBorder), kd];

                                    // Update kernel based on gradient
                                    _kernels[f, kw, kh, kd] -= weightGradientWHF * learningRate;

                                    nextGradients[w - (kernelBorder - padding), h - (kernelBorder - padding), kd] += weightGradientWHF;
                                }
                            }
                        }
                    }
                }
            }

            //// Gradients for the biases of each neuron
            //for (int n = 0; n < _numberOfFeatures; n++)
            //{
            //    // Update biases based on gradients
            //    _biases[n] -= gradients[n] * learningRate;
            //}

            return nextGradients;
        }

        private double ClipGradient(double gradient)
        {
            return Math.Min(Math.Max(gradient, -1), 1);
        }
    }
}