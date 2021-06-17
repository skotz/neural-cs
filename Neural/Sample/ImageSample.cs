using System.Drawing;
using System.Text;

namespace Skotz.Neural.Sample
{
    public class ImageSample : ISample
    {
        public double[,,] Inputs { get; set; }

        public double[,,] Outputs { get; set; }

        public ISample Clone()
        {
            return new ImageSample
            {
                Inputs = Inputs?.Clone() as double[,,],
                Outputs = Outputs?.Clone() as double[,,]
            };
        }

        public ImageSample()
        {
        }

        public ImageSample(string image, bool[] classification)
        {
            using (var img = (Bitmap)Image.FromFile(image))
            {
                Inputs = new double[img.Width, img.Height, 3];

                for (var x = 0; x < img.Width; x++)
                {
                    for (var y = 0; y < img.Width; y++)
                    {
                        var pixel = img.GetPixel(x, y);

                        Inputs[x, y, 0] = (pixel.R - 127.5) / 127.5;
                        Inputs[x, y, 1] = (pixel.G - 127.5) / 127.5;
                        Inputs[x, y, 2] = (pixel.B - 127.5) / 127.5;
                    }
                }
            }

            Outputs = new double[classification.Length, 1, 1];

            for (int i = 0; i < classification.Length; i++)
            {
                Outputs[i, 0, 0] = classification[i] ? 1.0 : 0.0;
            }
        }

        public string OutputsToString()
        {
            var sb = new StringBuilder();

            for (int i = 0; i < Outputs.GetLength(0); i++)
            {
                sb.Append(Outputs[i, 0, 0]);

                if (i < Outputs.GetLength(0) - 1)
                {
                    sb.Append(", ");
                }
            }

            return sb.ToString();
        }
    }
}