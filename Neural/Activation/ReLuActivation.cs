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
    }
}