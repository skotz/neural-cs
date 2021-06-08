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
            var nn = new NeuralNetwork(new SquaredErrorLoss(), 0.1);
            nn.Add(new FullyConnectedLayer(3, 50, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(50, 20, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(20, 5, new LeakyReLuActivation()));

            var data1 = new Sample()
            {
                Inputs = new double[] { -1, 0.5, 1 },
                Outputs = new double[] { 0, 1, 0, 0, 0 }
            }; 
            var data2 = new Sample()
            {
                Inputs = new double[] { 1, 1, -0.25 },
                Outputs = new double[] { 0, 0, 1, 0, 0 }
            };

            var loss = 1.0;

            var result = nn.FeedForward(data1);
            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            result = nn.FeedForward(data2);
            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            for (int i = 0; i < 1000 && loss > 0.001; i++)
            {
                loss = nn.Train(new List<ISample> { data1, data2 });

                Console.WriteLine("Loss: " + loss);

                result = nn.FeedForward(data1);
                Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

                result = nn.FeedForward(data2);
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