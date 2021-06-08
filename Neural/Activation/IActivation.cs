using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Skotz.Neural.Activation
{
    public interface IActivation
    {
        double Run(double value);

        double Derivative(double value);
    }
}
