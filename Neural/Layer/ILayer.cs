namespace Skotz.Neural.Layer
{
    public interface ILayer
    {
        double[,,] FeedForward(double[,,] values);

        double[,,] Backpropagate(double[,,] gradients, double learningRate);
    }
}