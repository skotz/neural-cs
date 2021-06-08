namespace Skotz.Neural.Layer
{
    public interface ILayer
    {
        int Size { get; }

        double[] FeedForward(double[] values);

        double[] Backpropagate(double[] gradients, double learningRate);
    }
}