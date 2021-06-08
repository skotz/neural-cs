using System;

namespace Skotz.Neural.Loss
{
    public class SquaredErrorLoss : ILoss
    {
        public double[] Gradients(double[] output, double[] expected)
        {
            var loss = new double[output.Length];

            for (int i = 0; i < output.Length; i++)
            {
                // dL/d[i] = 2 * (o[i] - e[i]) / 2 = o[i] - e[i]
                loss[i] = output[i] - expected[i];
            }

            return loss;
        }

        public double Total(double[] output, double[] expected)
        {
            var loss = 0.0;

            for (int i = 0; i < output.Length; i++)
            {
                loss += Math.Pow(output[i] - expected[i], 2) / 2;
            }

            return loss;
        }
    }
}