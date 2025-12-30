using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private sealed class LayerBuffers
        {
            public readonly double[][] activations;
            public readonly double[][] preActivations;
            public readonly double[][] deltas;

            public LayerBuffers(int[] structure)
            {
                int layers = structure.Length;
                activations = new double[layers][];
                preActivations = new double[layers][];
                deltas = new double[layers][];

                for (int layerIndex = 0; layerIndex < layers; layerIndex++)
                {
                    int size = structure[layerIndex];
                    activations[layerIndex] = new double[size];
                    preActivations[layerIndex] = new double[size];
                    deltas[layerIndex] = new double[size];
                }
            }

            public void ClearDeltas()
            {
                for (int layerIndex = 0; layerIndex < deltas.Length; layerIndex++)
                    Array.Clear(deltas[layerIndex], 0, deltas[layerIndex].Length);
            }
        }

        private sealed class ThreadState
        {
            public readonly double[][][] grad;
            public readonly LayerBuffers buffers;
            public double lossSum;
            public int count;

            public ThreadState(double[][][] grad, LayerBuffers buffers)
            {
                this.grad = grad;
                this.buffers = buffers;
                lossSum = 0.0;
                count = 0;
            }
        }

        private readonly int[] structure;
        private readonly int layersCount;

        private readonly double[][][] weights;
        private readonly double[][][] gradWeights;

        private readonly double[][][] adamM;
        private readonly double[][][] adamV;

        private readonly Random random = new Random();
        private readonly LayerBuffers mainBuffers;

        private double learningRate = 0.003;
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double adamEps = 1e-8;

        private double l2 = 0.0;
        private double gradClip = 5.0;

        private int batchSize = 16;
        private long adamStep = 0;

        private bool hasScaler = false;
        private readonly double[] inputMean;
        private readonly double[] inputInvStd;

        public Stopwatch stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            this.structure = (int[])structure.Clone();
            layersCount = this.structure.Length;

            mainBuffers = new LayerBuffers(this.structure);

            int inputSize = this.structure[0];
            inputMean = new double[inputSize];
            inputInvStd = new double[inputSize];

            int wLayers = layersCount - 1;
            weights = new double[wLayers][][];
            gradWeights = new double[wLayers][][];

            adamM = new double[wLayers][][];
            adamV = new double[wLayers][][];

            for (int layerIndex = 1; layerIndex < layersCount; layerIndex++)
            {
                int prevCount = this.structure[layerIndex - 1];
                int curCount = this.structure[layerIndex];

                weights[layerIndex - 1] = AllocateLayer(curCount, prevCount + 1);
                gradWeights[layerIndex - 1] = AllocateLayer(curCount, prevCount + 1);

                adamM[layerIndex - 1] = AllocateLayer(curCount, prevCount + 1);
                adamV[layerIndex - 1] = AllocateLayer(curCount, prevCount + 1);

                InitLayer(weights[layerIndex - 1], prevCount, curCount);
            }
        }

        // Evaluate: вычисляет средний cross-entropy loss и accuracy (по argmax) на переданном наборе
        public void Evaluate(SamplesSet testSet, out double loss, out double accuracy)
        {
            loss = 0.0;
            accuracy = 0.0;

            if (testSet == null || testSet.Count == 0)
            {
                loss = double.PositiveInfinity;
                accuracy = 0.0;
                return;
            }

            int correct = 0;
            int n = testSet.Count;

            for (int i = 0; i < n; i++)
            {
                var s = testSet[i];
                double[] pred = Compute(s.input); // использует уже реализацию Forward + Softmax
                loss += CrossEntropy(pred, s.Output);

                // argmax для предсказания и для эталона
                int predIdx = 0;
                double predVal = pred[0];
                for (int k = 1; k < pred.Length; k++)
                    if (pred[k] > predVal) { predVal = pred[k]; predIdx = k; }

                int trueIdx = 0;
                double trueVal = s.Output[0];
                for (int k = 1; k < s.Output.Length; k++)
                    if (s.Output[k] > trueVal) { trueVal = s.Output[k]; trueIdx = k; }

                if (predIdx == trueIdx) correct++;
            }

            loss /= n;
            accuracy = (double)correct / n;
        }

        public void TrainWithValidation(SamplesSet fullSet, double trainRatio, int epochsCount, double acceptableError, bool parallel, out double testLoss, out double testAccuracy, int? seed = null)
        {
            if (fullSet == null) throw new ArgumentNullException(nameof(fullSet));
            if (trainRatio < 0.0 || trainRatio > 1.0) throw new ArgumentOutOfRangeException(nameof(trainRatio));

            fullSet.Split(trainRatio, out SamplesSet trainSet, out SamplesSet testSet, seed);

            if (trainSet.Count == 0)
            {
                // ничего не обучаем — сразу оцениваем на тесте (если есть)
                if (testSet.Count == 0)
                {
                    testLoss = double.PositiveInfinity;
                    testAccuracy = 0.0;
                    return;
                }

                // подготовим скейлер по тесту (хотя обычно масштабируем по train)
                FitScalerFromProvider(testSet.Count, i => testSet[i].input);
                Evaluate(testSet, out testLoss, out testAccuracy);
                return;
            }

            // Фитируем скейлер строго по train
            FitScalerFromProvider(trainSet.Count, i => trainSet[i].input);

            // Используем существующий TrainOnProvider для обучения на trainSet
            TrainOnProvider(trainSet.Count, i => trainSet[i], epochsCount, acceptableError, parallel);

            // Оцениваем на тесте
            Evaluate(testSet, out testLoss, out testAccuracy);
        }

        public void SetBatchSize(int newBatchSize)
        {
            if (newBatchSize < 1) newBatchSize = 1;
            batchSize = newBatchSize;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;
            double loss = double.PositiveInfinity;

            while (loss > acceptableError)
            {
                iterations++;
                loss = TrainOnce(sample);
                if (iterations > 20000) break;
            }

            return iterations;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            if (samplesSet == null || samplesSet.Count == 0) return double.PositiveInfinity;

            FitScalerFromProvider(samplesSet.Count, i => samplesSet[i].input);
            return TrainOnProvider(samplesSet.Count, i => samplesSet[i], epochsCount, acceptableError, parallel);
        }

        public double TrainOnProvider(int sampleCount, Func<int, Sample> sampleProvider, int epochsCount, double acceptableError, bool parallel)
        {
            if (sampleCount <= 0) return double.PositiveInfinity;

            stopWatch.Restart();

            double error = double.PositiveInfinity;
            int[] indices = Enumerable.Range(0, sampleCount).ToArray();

            for (int epochIndex = 1; epochIndex <= epochsCount; epochIndex++)
            {
                Shuffle(indices);

                double epochLoss = 0.0;
                int epochSeen = 0;

                for (int start = 0; start < indices.Length; start += batchSize)
                {
                    int end = System.Math.Min(indices.Length, start + batchSize);
                    int currentBatch = end - start;
                    if (currentBatch <= 0) continue;

                    Sample[] batch = new Sample[currentBatch];
                    for (int i = 0; i < currentBatch; i++)
                        batch[i] = sampleProvider(indices[start + i]);

                    double batchLoss = TrainOnBatch(batch, parallel);

                    epochLoss += batchLoss * currentBatch;
                    epochSeen += currentBatch;

                    OnTrainProgress((epochIndex - 1 + (start + currentBatch) * 1.0 / indices.Length) / epochsCount, epochLoss / System.Math.Max(1, epochSeen), stopWatch.Elapsed);
                }

                error = epochLoss / System.Math.Max(1, epochSeen);
                OnTrainProgress(epochIndex * 1.0 / epochsCount, error, stopWatch.Elapsed);

                if (error <= acceptableError) break;
            }

            OnTrainProgress(1.0, error, stopWatch.Elapsed);
            stopWatch.Stop();
            return error;
        }

        public void FitScalerFromProvider(int sampleCount, Func<int, double[]> inputProvider)
        {
            int inputSize = structure[0];
            int n = System.Math.Max(1, sampleCount);

            Array.Clear(inputMean, 0, inputMean.Length);
            Array.Clear(inputInvStd, 0, inputInvStd.Length);

            for (int i = 0; i < sampleCount; i++)
            {
                double[] x = inputProvider(i);
                for (int j = 0; j < inputSize; j++)
                    inputMean[j] += x[j];
            }

            double invN = 1.0 / n;
            for (int j = 0; j < inputSize; j++)
                inputMean[j] *= invN;

            for (int i = 0; i < sampleCount; i++)
            {
                double[] x = inputProvider(i);
                for (int j = 0; j < inputSize; j++)
                {
                    double d = x[j] - inputMean[j];
                    inputInvStd[j] += d * d;
                }
            }

            for (int j = 0; j < inputSize; j++)
            {
                double var = inputInvStd[j] * invN;
                double std = System.Math.Sqrt(System.Math.Max(1e-12, var));
                inputInvStd[j] = 1.0 / std;
            }

            hasScaler = true;
        }

        public double TrainOnBatch(Sample[] batch, bool parallel)
        {
            int currentBatch = batch.Length;
            if (currentBatch <= 0) return 0.0;

            ZeroGradients();

            double batchLoss = 0.0;
            int batchCount = 0;

            if (parallel && currentBatch >= 4)
            {
                object mergeLock = new object();

                Parallel.For(
                    0,
                    currentBatch,
                    () => CreateThreadState(),
                    (offset, loopState, state) =>
                    {
                        double localLoss = Backprop(batch[offset], state.grad, state.buffers);
                        state.lossSum += localLoss;
                        state.count++;
                        return state;
                    },
                    state =>
                    {
                        lock (mergeLock)
                        {
                            AddGradients(state.grad);
                            batchLoss += state.lossSum;
                            batchCount += state.count;
                        }
                    }
                );
            }
            else
            {
                for (int i = 0; i < currentBatch; i++)
                    batchLoss += Backprop(batch[i], gradWeights, mainBuffers);

                batchCount = currentBatch;
            }

            if (l2 > 0) AddL2Gradients(batchCount);
            if (gradClip > 0) ClipGradients();

            ApplyAdamUpdate(batchCount);

            return batchLoss / System.Math.Max(1, batchCount);
        }

        protected override double[] Compute(double[] input)
        {
            Forward(input, mainBuffers);
            int outSize = structure[layersCount - 1];

            double[] output = new double[outSize];
            Array.Copy(mainBuffers.activations[layersCount - 1], output, outSize);
            return output;
        }

        public void Save(string path)
        {
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(0x4E4E5331);
                bw.Write(1);

                bw.Write(structure.Length);
                for (int i = 0; i < structure.Length; i++) bw.Write(structure[i]);

                bw.Write(hasScaler);

                bw.Write(inputMean.Length);
                for (int i = 0; i < inputMean.Length; i++) bw.Write(inputMean[i]);
                for (int i = 0; i < inputInvStd.Length; i++) bw.Write(inputInvStd[i]);

                bw.Write(adamStep);

                Write3D(bw, weights);
                Write3D(bw, adamM);
                Write3D(bw, adamV);

                bw.Write(learningRate);
                bw.Write(beta1);
                bw.Write(beta2);
                bw.Write(adamEps);
                bw.Write(l2);
                bw.Write(gradClip);
                bw.Write(batchSize);
            }
        }

        public void Load(string path)
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var br = new BinaryReader(fs))
            {
                int magic = br.ReadInt32();
                if (magic != 0x4E4E5331) throw new InvalidDataException("Bad model file");

                int version = br.ReadInt32();
                if (version != 1) throw new InvalidDataException("Unsupported model version");

                int structLen = br.ReadInt32();
                if (structLen != structure.Length) throw new InvalidDataException("Structure mismatch");

                for (int i = 0; i < structLen; i++)
                    if (br.ReadInt32() != structure[i])
                        throw new InvalidDataException("Structure mismatch");

                hasScaler = br.ReadBoolean();

                int scalerLen = br.ReadInt32();
                if (scalerLen != inputMean.Length) throw new InvalidDataException("Scaler size mismatch");

                for (int i = 0; i < inputMean.Length; i++) inputMean[i] = br.ReadDouble();
                for (int i = 0; i < inputInvStd.Length; i++) inputInvStd[i] = br.ReadDouble();

                adamStep = br.ReadInt64();

                Read3DInto(br, weights);
                Read3DInto(br, adamM);
                Read3DInto(br, adamV);

                learningRate = br.ReadDouble();
                beta1 = br.ReadDouble();
                beta2 = br.ReadDouble();
                adamEps = br.ReadDouble();
                l2 = br.ReadDouble();
                gradClip = br.ReadDouble();
                batchSize = br.ReadInt32();
            }
        }

        private double TrainOnce(Sample sample)
        {
            ZeroGradients();

            double loss = Backprop(sample, gradWeights, mainBuffers);

            if (l2 > 0) AddL2Gradients(1);
            if (gradClip > 0) ClipGradients();

            ApplyAdamUpdate(1);
            return loss;
        }

        private static double[][] AllocateLayer(int rows, int cols)
        {
            double[][] layer = new double[rows][];
            for (int i = 0; i < rows; i++)
                layer[i] = new double[cols];
            return layer;
        }

        private void InitLayer(double[][] layerWeights, int prevCount, int curCount)
        {
            double limit = System.Math.Sqrt(6.0 / (prevCount + curCount));
            for (int neuronIndex = 0; neuronIndex < curCount; neuronIndex++)
            {
                double[] w = layerWeights[neuronIndex];
                for (int weightIndex = 0; weightIndex < w.Length; weightIndex++)
                    w[weightIndex] = NextUniform(-limit, limit);
            }
        }

        private void Forward(double[] input, LayerBuffers buffers)
        {
            if (input.Length != structure[0])
                throw new ArgumentException("Input size mismatch");

            WriteInputActivations(input, buffers.activations[0]);

            for (int layerIndex = 1; layerIndex < layersCount; layerIndex++)
            {
                int prevCount = structure[layerIndex - 1];
                int curCount = structure[layerIndex];

                double[] prevA = buffers.activations[layerIndex - 1];
                double[] z = buffers.preActivations[layerIndex];
                double[] a = buffers.activations[layerIndex];

                double[][] layerW = weights[layerIndex - 1];

                for (int i = 0; i < curCount; i++)
                {
                    double[] w = layerW[i];
                    double sum = w[prevCount];

                    for (int j = 0; j < prevCount; j++)
                        sum += w[j] * prevA[j];

                    z[i] = sum;
                }

                if (layerIndex == layersCount - 1)
                {
                    SoftmaxInto(z, a);
                }
                else
                {
                    for (int i = 0; i < curCount; i++)
                        a[i] = System.Math.Max(0.0, z[i]);
                }
            }
        }

        private void WriteInputActivations(double[] input, double[] dst)
        {
            if (hasScaler)
            {
                for (int i = 0; i < dst.Length; i++)
                    dst[i] = (input[i] - inputMean[i]) * inputInvStd[i];
                return;
            }

            double max = 0.0;
            for (int i = 0; i < input.Length; i++)
                if (input[i] > max) max = input[i];

            double scale = max > 1.5 ? 1.0 / 200.0 : 1.0;
            for (int i = 0; i < dst.Length; i++)
                dst[i] = input[i] * scale;
        }

        private double Backprop(Sample sample, double[][][] targetGrad, LayerBuffers buffers)
        {
            Forward(sample.input, buffers);

            int last = layersCount - 1;
            double[] yPred = buffers.activations[last];
            double[] yTrue = sample.Output;

            double loss = CrossEntropy(yPred, yTrue);

            buffers.ClearDeltas();

            for (int i = 0; i < structure[last]; i++)
                buffers.deltas[last][i] = yPred[i] - yTrue[i];

            for (int layerIndex = last - 1; layerIndex >= 1; layerIndex--)
            {
                int curCount = structure[layerIndex];
                int nextCount = structure[layerIndex + 1];

                double[] curDelta = buffers.deltas[layerIndex];
                double[] nextDelta = buffers.deltas[layerIndex + 1];
                double[] curA = buffers.activations[layerIndex];

                double[][] nextW = weights[layerIndex];

                for (int i = 0; i < curCount; i++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < nextCount; k++)
                        sum += nextDelta[k] * nextW[k][i];

                    double a = curA[i];
                    curDelta[i] = (a > 0.0 ? 1.0 : 0.0) * sum;
                }
            }

            for (int layerIndex = 1; layerIndex < layersCount; layerIndex++)
            {
                int prevCount = structure[layerIndex - 1];
                int curCount = structure[layerIndex];

                double[] prevA = buffers.activations[layerIndex - 1];
                double[] delta = buffers.deltas[layerIndex];

                double[][] layerGrad = targetGrad[layerIndex - 1];

                for (int i = 0; i < curCount; i++)
                {
                    double d = delta[i];
                    double[] g = layerGrad[i];

                    for (int j = 0; j < prevCount; j++)
                        g[j] += d * prevA[j];

                    g[prevCount] += d;
                }
            }

            return loss;
        }

        private static void SoftmaxInto(double[] logits, double[] output)
        {
            double max = logits[0];
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > max) max = logits[i];

            double sum = 0.0;
            for (int i = 0; i < logits.Length; i++)
            {
                double e = System.Math.Exp(logits[i] - max);
                output[i] = e;
                sum += e;
            }

            double invSum = 1.0 / System.Math.Max(1e-12, sum);
            for (int i = 0; i < output.Length; i++)
                output[i] *= invSum;
        }

        private static double CrossEntropy(double[] yPred, double[] yTrue)
        {
            double loss = 0.0;
            for (int i = 0; i < yPred.Length; i++)
            {
                double p = System.Math.Max(1e-12, yPred[i]);
                loss -= yTrue[i] * System.Math.Log(p);
            }
            return loss;
        }

        private void ZeroGradients()
        {
            for (int l = 0; l < gradWeights.Length; l++)
                for (int i = 0; i < gradWeights[l].Length; i++)
                    Array.Clear(gradWeights[l][i], 0, gradWeights[l][i].Length);
        }

        private ThreadState CreateThreadState()
        {
            double[][][] grad = new double[gradWeights.Length][][];
            for (int l = 0; l < grad.Length; l++)
                grad[l] = AllocateLayer(gradWeights[l].Length, gradWeights[l][0].Length);

            return new ThreadState(grad, new LayerBuffers(structure));
        }

        private void AddGradients(double[][][] other)
        {
            for (int l = 0; l < gradWeights.Length; l++)
            {
                for (int i = 0; i < gradWeights[l].Length; i++)
                {
                    double[] dst = gradWeights[l][i];
                    double[] src = other[l][i];
                    for (int j = 0; j < dst.Length; j++)
                        dst[j] += src[j];
                }
            }
        }

        private void AddL2Gradients(int batchCount)
        {
            double scale = l2 / System.Math.Max(1, batchCount);

            for (int l = 0; l < weights.Length; l++)
            {
                for (int i = 0; i < weights[l].Length; i++)
                {
                    double[] w = weights[l][i];
                    double[] g = gradWeights[l][i];

                    for (int j = 0; j < w.Length; j++)
                        g[j] += scale * w[j];
                }
            }
        }

        private void ClipGradients()
        {
            double sumSq = 0.0;

            for (int l = 0; l < gradWeights.Length; l++)
                for (int i = 0; i < gradWeights[l].Length; i++)
                    for (int j = 0; j < gradWeights[l][i].Length; j++)
                        sumSq += gradWeights[l][i][j] * gradWeights[l][i][j];

            double norm = System.Math.Sqrt(sumSq);
            if (norm <= gradClip || norm <= 1e-12) return;

            double scale = gradClip / norm;

            for (int l = 0; l < gradWeights.Length; l++)
                for (int i = 0; i < gradWeights[l].Length; i++)
                    for (int j = 0; j < gradWeights[l][i].Length; j++)
                        gradWeights[l][i][j] *= scale;
        }

        private void ApplyAdamUpdate(int batchCount)
        {
            adamStep++;

            double invBatch = 1.0 / System.Math.Max(1, batchCount);
            double b1t = System.Math.Pow(beta1, adamStep);
            double b2t = System.Math.Pow(beta2, adamStep);
            double lrT = learningRate * System.Math.Sqrt(1.0 - b2t) / (1.0 - b1t);

            for (int l = 0; l < weights.Length; l++)
            {
                for (int i = 0; i < weights[l].Length; i++)
                {
                    double[] w = weights[l][i];
                    double[] g = gradWeights[l][i];
                    double[] m = adamM[l][i];
                    double[] v = adamV[l][i];

                    for (int j = 0; j < w.Length; j++)
                    {
                        double grad = g[j] * invBatch;

                        m[j] = beta1 * m[j] + (1.0 - beta1) * grad;
                        v[j] = beta2 * v[j] + (1.0 - beta2) * grad * grad;

                        double denom = System.Math.Sqrt(v[j]) + adamEps;
                        w[j] -= lrT * m[j] / denom;
                    }
                }
            }
        }

        private void Shuffle(int[] indices)
        {
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                int t = indices[i];
                indices[i] = indices[j];
                indices[j] = t;
            }
        }

        private double NextUniform(double min, double max)
        {
            return min + (max - min) * random.NextDouble();
        }

        private static void Write3D(BinaryWriter bw, double[][][] arr)
        {
            bw.Write(arr.Length);
            for (int l = 0; l < arr.Length; l++)
            {
                bw.Write(arr[l].Length);
                for (int i = 0; i < arr[l].Length; i++)
                {
                    bw.Write(arr[l][i].Length);
                    for (int j = 0; j < arr[l][i].Length; j++)
                        bw.Write(arr[l][i][j]);
                }
            }
        }

        private static void Read3DInto(BinaryReader br, double[][][] dst)
        {
            int lCount = br.ReadInt32();
            if (lCount != dst.Length) throw new InvalidDataException("Layer count mismatch");

            for (int l = 0; l < dst.Length; l++)
            {
                int rows = br.ReadInt32();
                if (rows != dst[l].Length) throw new InvalidDataException("Rows mismatch");

                for (int i = 0; i < dst[l].Length; i++)
                {
                    int cols = br.ReadInt32();
                    if (cols != dst[l][i].Length) throw new InvalidDataException("Cols mismatch");

                    for (int j = 0; j < dst[l][i].Length; j++)
                        dst[l][i][j] = br.ReadDouble();
                }
            }
        }
    }
}
