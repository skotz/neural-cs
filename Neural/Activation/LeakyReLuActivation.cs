using System;

namespace Skotz.Neural.Activation
{
    public class LeakyReLuActivation : IActivation
    {
        private double _factor = 0.01;

        public double Run(double value)
        {
            return Math.Max(_factor * value, value);
        }

        public double Derivative(double value)
        {
            return value > 0 ? 1 : _factor;
        }
    }
}