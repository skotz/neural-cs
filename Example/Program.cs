using Skotz.Neural.Activation;
using Skotz.Neural.Layer;
using Skotz.Neural.Loss;
using Skotz.Neural.Network;
using Skotz.Neural.Sample;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Example
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var nn = new NeuralNetwork(new SquaredErrorLoss());
            nn.Add(new FullyConnectedLayer(3, 10, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(10, 5, new LeakyReLuActivation()));

            var data = new Sample()
            {
                Inputs = new double[] { -1, 0.5, 1 },
                Outputs = new double[] { 0, 1, 0, 0, 0 }
            };

            var result = nn.FeedForward(data);
            var loss = 1.0;

            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            for (int i = 0; i < 100 && loss > 0.001; i++)
            {
                loss = nn.Train(new List<ISample> { data });

                Console.WriteLine("Loss: " + loss);

                result = nn.FeedForward(data);

                Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));
            }

            Console.ReadKey();
        }
    }

    public class Sample : ISample
    {
        public double[] Inputs { get; set; }

        public double[] Outputs { get; set; }

        public ISample Clone()
        {
            return new Sample
            {
                Inputs = Inputs?.Clone() as double[],
                Outputs = Outputs?.Clone() as double[]
            };
        }
    }
}