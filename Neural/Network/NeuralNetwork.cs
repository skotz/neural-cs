using Skotz.Neural.Layer;
using Skotz.Neural.Loss;
using Skotz.Neural.Sample;
using System;
using System.Collections.Generic;

namespace Skotz.Neural.Network
{
    public class NeuralNetwork
    {
        private List<ILayer> _layers;
        private ILoss _lossFunction;

        public NeuralNetwork(ILoss lossFunction)
        {
            _layers = new List<ILayer>();
            _lossFunction = lossFunction;
        }

        public void Add(ILayer layer)
        {
            _layers.Add(layer);
        }

        public ISample FeedForward(ISample data)
        {
            var result = data.Clone();
            var activations = result.Inputs;

            foreach (var layer in _layers)
            {
                activations = layer.FeedForward(activations);
            }

            result.Outputs = activations;

            return result;
        }

        public double Train(List<ISample> samples)
        {
            var totalLoss = 0.0;

            foreach (var sample in samples)
            {
                var output = FeedForward(sample);

                // Output layer gradients
                var gradients = _lossFunction.Gradients(output.Outputs, sample.Outputs);

                // Backpropagation
                for (int i = _layers.Count - 1; i >= 0; i--)
                {
                    gradients = _layers[i].Backpropagate(gradients);
                }

                totalLoss += _lossFunction.Total(output.Outputs, sample.Outputs);
            }

            return totalLoss;
        }
    }
}