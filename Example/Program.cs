using Skotz.Neural.Activation;
using Skotz.Neural.Layer;
using Skotz.Neural.Loss;
using Skotz.Neural.Network;
using Skotz.Neural.Sample;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Example
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var nn = new NeuralNetwork(new SquaredErrorLoss(), 0.0001);
            nn.Add(new FullyConnectedLayer(2, 5, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(5, 5, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(5, 1, new LeakyReLuActivation()));

            var data1 = new Sample()
            {
                Inputs = new double[] { 0, 0 },
                Outputs = new double[] { 0 }
            };
            var data2 = new Sample()
            {
                Inputs = new double[] { 0, 1 },
                Outputs = new double[] { 1 }
            };
            var data3 = new Sample()
            {
                Inputs = new double[] { 1, 0 },
                Outputs = new double[] { 1 }
            };
            var data4 = new Sample()
            {
                Inputs = new double[] { 1, 1 },
                Outputs = new double[] { 0 }
            };

            var loss = 1.0;

            var result = nn.FeedForward(data1);
            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            result = nn.FeedForward(data2);
            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            result = nn.FeedForward(data3);
            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            result = nn.FeedForward(data4);
            Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

            using (var w = new StreamWriter("results.csv"))
            {
                for (int i = 0; i < 100000 && loss > 0.0001; i++)
                {
                    loss = nn.Train(new List<ISample> { data1, data2, data3, data4 });

                    Console.WriteLine($"Iteration {i + 1} \tLoss: {loss}");

                    w.WriteLine($"{loss}");

                    result = nn.FeedForward(data1);
                    Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

                    result = nn.FeedForward(data2);
                    Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

                    result = nn.FeedForward(data3);
                    Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));

                    result = nn.FeedForward(data4);
                    Console.WriteLine("Result: " + result.Outputs.Select(x => x.ToString("0.000")).Aggregate((c, n) => c + ", " + n));
                }
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