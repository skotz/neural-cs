using System;

namespace Skotz.Neural.Activation
{
    public class HyperbolicTangentActivation : IActivation
    {
        public double Derivative(double value)
        {
            return 1 - Math.Pow(Math.Tanh(value), 2);
        }

        public double Run(double value)
        {
            return Math.Tanh(value);
        }

        public double StandardDeviation(int inputs, int outputs)
        {
            // Xavier (Glorot) random weight initialization
            return Math.Sqrt(2.0 / (inputs + outputs));
        }
    }
}