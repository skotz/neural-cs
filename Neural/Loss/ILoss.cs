namespace Skotz.Neural.Loss
{
    public interface ILoss
    {
        double[] Gradients(double[] output, double[] expected);

        double Total(double[] output, double[] expected);
    }
}