using System.IO;

namespace Skotz.Neural.Layer
{
    public interface ILayer
    {
        double[,,] FeedForward(double[,,] values);

        double[,,] Backpropagate(double[,,] gradients, double learningRate);

        void Save(BinaryWriter writer);

        void Load(BinaryReader reader);
    }
}