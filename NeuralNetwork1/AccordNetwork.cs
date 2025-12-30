using System;
using System.Diagnostics;
using Accord.Neuro;
using Accord.Neuro.Learning;

namespace NeuralNetwork1
{
    public class AccordNetwork : BaseNetwork
    {
        private readonly ActivationNetwork network;
        private readonly ParallelResilientBackpropagationLearning teacher;

        private readonly int inputSize;
        private readonly int outputSize;

        // scaler
        private readonly double[] mean;
        private readonly double[] invStd;
        private bool hasScaler;

        private readonly Stopwatch sw = new Stopwatch();

        public AccordNetwork(int inputSize, int[] hiddenLayers, int outputSize)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;

            mean = new double[inputSize];
            invStd = new double[inputSize];

            // ===== ВАЖНО =====
            // Один activation на ВСЕ слои (так требует Accord)
            var activation = new BipolarSigmoidFunction();

            network = new ActivationNetwork(
                activation,
                inputSize,
                BuildLayerSizes(hiddenLayers, outputSize)
            );

            new NguyenWidrow(network).Randomize();

            teacher = new ParallelResilientBackpropagationLearning(network);
        }

        private static int[] BuildLayerSizes(int[] hidden, int output)
        {
            int[] sizes = new int[hidden.Length + 1];
            Array.Copy(hidden, sizes, hidden.Length);
            sizes[sizes.Length - 1] = output;
            return sizes;
        }

        // =========================
        // SCALER (как в StudentNetwork)
        // =========================
        private void FitScaler(SamplesSet set)
        {
            Array.Clear(mean, 0, mean.Length);
            Array.Clear(invStd, 0, invStd.Length);

            for (int i = 0; i < set.Count; i++)
                for (int j = 0; j < inputSize; j++)
                    mean[j] += set[i].input[j];

            for (int j = 0; j < inputSize; j++)
                mean[j] /= set.Count;

            for (int i = 0; i < set.Count; i++)
                for (int j = 0; j < inputSize; j++)
                {
                    double d = set[i].input[j] - mean[j];
                    invStd[j] += d * d;
                }

            for (int j = 0; j < inputSize; j++)
                invStd[j] = 1.0 / Math.Sqrt(Math.Max(1e-12, invStd[j] / set.Count));

            hasScaler = true;
        }

        private void Scale(double[] src, double[] dst)
        {
            for (int i = 0; i < src.Length; i++)
                dst[i] = (src[i] - mean[i]) * invStd[i];
        }

        // =========================
        protected override double[] Compute(double[] input)
        {
            double[] x = new double[inputSize];
            if (hasScaler) Scale(input, x);
            else Array.Copy(input, x, inputSize);

            return network.Compute(x);
        }

        private static double[] Softmax(double[] z)
        {
            double max = z[0];
            for (int i = 1; i < z.Length; i++)
                if (z[i] > max) max = z[i];

            double sum = 0;
            double[] e = new double[z.Length];

            for (int i = 0; i < z.Length; i++)
            {
                e[i] = Math.Exp(z[i] - max);
                sum += e[i];
            }

            for (int i = 0; i < z.Length; i++)
                e[i] /= sum;

            return e;
        }

        // =========================
        public override double TrainOnDataSet(
            SamplesSet set,
            int epochs,
            double acceptableError,
            bool parallel)
        {
            FitScaler(set);

            double[][] x = new double[set.Count][];
            double[][] y = new double[set.Count][];

            for (int i = 0; i < set.Count; i++)
            {
                x[i] = new double[inputSize];
                Scale(set[i].input, x[i]);
                y[i] = new double[outputSize];
                for (int k = 0; k < outputSize; k++)
                    y[i][k] = set[i].Output[k] > 0.5 ? 1.0 : -1.0;

            }

            double error = double.MaxValue;
            sw.Restart();

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                error = teacher.RunEpoch(x, y) / set.Count;

                OnTrainProgress(
                    epoch / (double)epochs,
                    error,
                    sw.Elapsed
                );

                if (error <= acceptableError)
                    break;
            }

            sw.Stop();
            return error;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            throw new NotSupportedException();
        }
    }
}
