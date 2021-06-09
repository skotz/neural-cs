using System;

namespace Skotz.Neural.Activation
{
    public class ReLuActivation : IActivation
    {
        public double Run(double value)
        {
            return Math.Max(0, value);
        }

        public double Derivative(double value)
        {
            return value > 0 ? 1 : 0;
        }

        public double StandardDeviation(int inputs, int outputs)
        {
            // "He" initialization
            return Math.Sqrt(4.0 / (inputs + outputs));
        }
    }
}