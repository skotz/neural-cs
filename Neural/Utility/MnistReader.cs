using Skotz.Neural.Sample;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;

namespace Skotz.Neural.Utility
{
    // http://yann.lecun.com/exdb/mnist/
    public class MnistReader
    {
        private const string TrainImages = "train-images.idx3-ubyte";
        private const string TrainLabels = "train-labels.idx1-ubyte";
        private const string TestImages = "t10k-images.idx3-ubyte";
        private const string TestLabels = "t10k-labels.idx1-ubyte";

        private string _folder;

        public MnistReader(string folder)
        {
            _folder = folder + (!folder.EndsWith("/") ? "/" : "");
        }

        public List<ISample> GetTrainingSamples()
        {
            return Read(_folder + TrainImages, _folder + TrainLabels);
        }

        public List<ISample> GetTestingSamples()
        {
            return Read(_folder + TestImages, _folder + TestLabels);
        }

        public void WriteToFolder(string location)
        {
            if (!Directory.Exists(location + "\\train"))
            {
                Directory.CreateDirectory(location + "\\train");
            }
            if (!Directory.Exists(location + "\\test"))
            {
                Directory.CreateDirectory(location + "\\test");
            }

            Extract($"{location}\\train", GetTrainingSamples());
            Extract($"{location}\\test", GetTestingSamples());
        }

        private static void Extract(string location, List<ISample> samples)
        {
            Parallel.ForEach(samples, sample =>
            {
                var folder = $"{location}\\{(sample as ImageSample)?.OutputToIndex()}";
                if (!Directory.Exists(folder))
                {
                    Directory.CreateDirectory(folder);
                }

                var width = sample.Inputs.GetLength(0);
                var height = sample.Inputs.GetLength(1);

                var file = $"{folder}\\{Guid.NewGuid()}.png";

                using (var image = new Bitmap(width, height))
                {
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            var pixel = (int)(sample.Inputs[x, y, 0] * 127.5 + 127.5);
                            image.SetPixel(x, y, Color.FromArgb(pixel, pixel, pixel));
                        }
                    }

                    image.Save(file);
                }
            });
        }

        private List<ISample> Read(string imagesPath, string labelsPath)
        {
            var samples = new List<ISample>();
            using (var labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open)))
            using (var images = new BinaryReader(new FileStream(imagesPath, FileMode.Open)))
            {
                _ = images.ReadBigInt32();

                var numberOfImages = images.ReadBigInt32();
                var width = images.ReadBigInt32();
                var height = images.ReadBigInt32();

                _ = labels.ReadBigInt32();
                _ = labels.ReadBigInt32();

                for (int i = 0; i < numberOfImages; i++)
                {
                    var bytes = images.ReadBytes(width * height);
                    var inputVolume = new double[width, height, 1];

                    for (int j = 0; j < width; j++)
                    {
                        for (int k = 0; k < height; k++)
                        {
                            inputVolume[j, k, 0] = (bytes[k * height + j] - 127.5) / 127.5;
                        }
                    }

                    var label = new double[10, 1, 1];
                    label[labels.ReadByte(), 0, 0] = 1;

                    samples.Add(new ImageSample
                    {
                        Inputs = inputVolume,
                        Outputs = label
                    });
                }

                return samples;
            }
        }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}