namespace NeuralNetwork1
{
    public sealed class Sample
    {
        public readonly double[] input;
        public readonly double[] Output;

        public Sample(double[] input, double[] output)
        {
            this.input = input;
            Output = output;
        }
    }
}
