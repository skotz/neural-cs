namespace Skotz.Neural.Sample
{
    public interface ISample
    {
        double[,,] Inputs { get; set; }

        double[,,] Outputs { get; set; }

        ISample Clone();

        string OutputsToString();
    }
}