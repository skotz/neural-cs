using Skotz.Neural.Activation;
using Skotz.Neural.Layer;
using Skotz.Neural.Loss;
using Skotz.Neural.Network;
using Skotz.Neural.Sample;
using Skotz.Neural.Utility;
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
            //RunXorTest();

            //RunConvTest();

            RunMnistTest2();

            Console.ReadKey();
        }

        private static void RunMnistTest2()
        {
            var nn = new NeuralNetwork(new SquaredErrorLoss(), 0.0001);
            nn.Add(new ConvolutionLayer(28, 28, 1, 3, 32, 4, new LeakyReLuActivation()));
            //nn.Add(new ConvolutionLayer(26, 26, 10, 3, 5, 1, new LeakyReLuActivation()));
            nn.Add(new FlattenLayer(7, 7, 32));
            nn.Add(new FullyConnectedLayer(7 * 7 * 32, 10, new LeakyReLuActivation()));
            //nn.Add(new FullyConnectedLayer(100, 10, new LeakyReLuActivation()));

            if (File.Exists("nn2.dat"))
            {
                using (var save = new FileStream("nn2.dat", FileMode.Open))
                {
                    nn.Load(save);
                }
                Console.WriteLine("Loaded model");
            }

            var reader = new MnistReader("j:\\mnist");
            var training = reader.GetTrainingSamples();
            var testing = reader.GetTestingSamples();

            // Only use a subset of the data
            testing.Shuffle();
            testing = testing.Take(100).ToList();

            var loss = 1.0;

            using (var w = new StreamWriter("results-mnist-cnn-2.csv"))
            {
                w.WriteLine("iteration,trainLoss,testLoss,testRate");

                for (int i = 0; i < 100 && loss > 0.05; i++)
                {
                    training.Shuffle();
                    var subset = training.Take(100).ToList();

                    loss = nn.Train(subset);

                    var test = nn.TestLoss(testing);
                    var rate = nn.TestRate<ImageSample>(testing, (c, n) => c.OutputToIndex() == n.OutputToIndex());

                    Console.WriteLine($"Iteration {i}\tLoss {loss}\tTest {test}\tCorrect {rate}");

                    w.WriteLine($"{i},{loss},{test},{rate}");

                    using (var save = new FileStream("nn2.dat", FileMode.Create))
                    {
                        nn.Save(save);
                    }
                }
            }
        }

        private static void RunMnistTest()
        {
            var nn = new NeuralNetwork(new SquaredErrorLoss(), 0.0001);
            nn.Add(new ConvolutionLayer(28, 28, 1, 3, 32, 1, new LeakyReLuActivation()));
            //nn.Add(new ConvolutionLayer(26, 26, 10, 3, 5, 1, new LeakyReLuActivation()));
            nn.Add(new FlattenLayer(26, 26, 32));
            nn.Add(new FullyConnectedLayer(26 * 26 * 32, 10, new LeakyReLuActivation()));
            //nn.Add(new FullyConnectedLayer(100, 10, new LeakyReLuActivation()));

            if (File.Exists("nn.dat"))
            {
                using (var save = new FileStream("nn.dat", FileMode.Open))
                {
                    nn.Load(save);
                }
                Console.WriteLine("Loaded model");
            }

            var reader = new MnistReader("j:\\mnist");
            var training = reader.GetTrainingSamples();
            var testing = reader.GetTestingSamples();

            // Only use a subset of the data
            testing.Shuffle();
            testing = testing.Take(100).ToList();

            var loss = 1.0;

            using (var w = new StreamWriter("results-mnist-cnn.csv"))
            {
                w.WriteLine("iteration,trainLoss,testLoss,testRate");

                for (int i = 0; i < 100 && loss > 0.05; i++)
                {
                    training.Shuffle();
                    var subset = training.Take(100).ToList();

                    loss = nn.Train(subset);

                    var test = nn.TestLoss(testing);
                    var rate = nn.TestRate<ImageSample>(testing, (c, n) => c.OutputToIndex() == n.OutputToIndex());

                    Console.WriteLine($"Iteration {i}\tLoss {loss}\tTest {test}\tCorrect {rate}");

                    w.WriteLine($"{i},{loss},{test},{rate}");

                    using (var save = new FileStream("nn.dat", FileMode.Create))
                    {
                        nn.Save(save);
                    }
                }
            }
        }

        private static void RunConvTest()
        {
            var nn = new NeuralNetwork(new SquaredErrorLoss(), 0.0001);
            nn.Add(new ConvolutionLayer(24, 24, 3, 3, 10, 1, new LeakyReLuActivation()));
            nn.Add(new ConvolutionLayer(22, 22, 10, 3, 5, 1, new LeakyReLuActivation()));
            nn.Add(new FlattenLayer(20, 20, 5));
            nn.Add(new FullyConnectedLayer(20 * 20 * 5, 100, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(100, 2, new LeakyReLuActivation()));

            var data1 = new ImageSample("../../../Data/Images/bear.png", new bool[] { true, false });
            var data2 = new ImageSample("../../../Data/Images/bird.png", new bool[] { false, true });

            var result1 = nn.FeedForward(data1);
            Console.WriteLine("Result: " + result1.OutputsToString());

            var result2 = nn.FeedForward(data1);
            Console.WriteLine("Result: " + result2.OutputsToString());

            var loss = 1.0;

            using (var w = new StreamWriter("results-cnn.csv"))
            {
                for (int i = 0; i < 100000 && loss > 0.01; i++)
                {
                    loss = nn.Train(new List<ISample> { data1, data2 });

                    Console.WriteLine($"Iteration {i + 1} \tLoss: {loss}");

                    w.WriteLine($"{loss}");

                    result1 = nn.FeedForward(data1);
                    Console.WriteLine("Result: " + result1.OutputsToString());

                    result2 = nn.FeedForward(data2);
                    Console.WriteLine("Result: " + result2.OutputsToString());
                }
            }
        }

        private static void RunXorTest()
        {
            var nn = new NeuralNetwork(new SquaredErrorLoss(), 0.01);
            nn.Add(new FullyConnectedLayer(2, 5, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(5, 5, new LeakyReLuActivation()));
            nn.Add(new FullyConnectedLayer(5, 1, new LeakyReLuActivation()));

            var data1 = new Sample()
            {
                Inputs = new double[,,] { { { 0 } }, { { 0 } } },
                Outputs = new double[,,] { { { 0 } } }
            };
            var data2 = new Sample()
            {
                Inputs = new double[,,] { { { 0 } }, { { 1 } } },
                Outputs = new double[,,] { { { 1 } } }
            };
            var data3 = new Sample()
            {
                Inputs = new double[,,] { { { 1 } }, { { 0 } } },
                Outputs = new double[,,] { { { 1 } } }
            };
            var data4 = new Sample()
            {
                Inputs = new double[,,] { { { 1 } }, { { 1 } } },
                Outputs = new double[,,] { { { 0 } } }
            };

            var loss = 1.0;

            var result = nn.FeedForward(data1);
            Console.WriteLine("Result: " + result.OutputsToString());

            result = nn.FeedForward(data2);
            Console.WriteLine("Result: " + result.OutputsToString());

            result = nn.FeedForward(data3);
            Console.WriteLine("Result: " + result.OutputsToString());

            result = nn.FeedForward(data4);
            Console.WriteLine("Result: " + result.OutputsToString());

            using (var w = new StreamWriter("results.csv"))
            {
                for (int i = 0; i < 100000 && loss > 0.001; i++)
                {
                    loss = nn.Train(new List<ISample> { data1, data2, data3, data4 });

                    Console.WriteLine($"Iteration {i + 1} \tLoss: {loss}");

                    w.WriteLine($"{loss}");

                    result = nn.FeedForward(data1);
                    Console.WriteLine("Result: " + result.OutputsToString());

                    result = nn.FeedForward(data2);
                    Console.WriteLine("Result: " + result.OutputsToString());

                    result = nn.FeedForward(data3);
                    Console.WriteLine("Result: " + result.OutputsToString());

                    result = nn.FeedForward(data4);
                    Console.WriteLine("Result: " + result.OutputsToString());
                }
            }
        }
    }

    public class Sample : ISample
    {
        public double[,,] Inputs { get; set; }

        public double[,,] Outputs { get; set; }

        public ISample Clone()
        {
            return new Sample
            {
                Inputs = Inputs?.Clone() as double[,,],
                Outputs = Outputs?.Clone() as double[,,]
            };
        }

        public string OutputsToString()
        {
            return Outputs[0, 0, 0].ToString("0.000");
        }
    }
}