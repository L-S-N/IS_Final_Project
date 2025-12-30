using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork1
{
    public sealed class SamplesSet
    {
        private readonly List<Sample> samples = new List<Sample>();

        public int Count => samples.Count;

        public Sample this[int index] => samples[index];

        public void Add(Sample sample)
        {
            samples.Add(sample);
        }

        /// <summary>
        /// Разбивает набор на обучающую и тестовую части.
        /// </summary>
        /// <param name="trainRatio">Доля элементов в обучающем наборе (0..1).</param>
        /// <param name="train">Результирующий обучающий набор.</param>
        /// <param name="test">Результирующий тестовый набор.</param>
        /// <param name="seed">Опциональный сид для детерминированного перемешивания. Если null — используется случайный сид.</param>
        public void Split(double trainRatio, out SamplesSet train, out SamplesSet test, int? seed = null)
        {
            if (trainRatio < 0.0 || trainRatio > 1.0)
                throw new ArgumentOutOfRangeException(nameof(trainRatio), "trainRatio must be between 0.0 and 1.0");

            train = new SamplesSet();
            test = new SamplesSet();

            if (Count == 0)
                return;

            int[] indices = Enumerable.Range(0, Count).ToArray();
            Shuffle(indices, seed);

            int trainCount = (int)Math.Round(trainRatio * Count);
            // Гарантируем, что границы корректны
            trainCount = Math.Max(0, Math.Min(Count, trainCount));

            for (int i = 0; i < trainCount; i++)
                train.Add(samples[indices[i]]);

            for (int i = trainCount; i < indices.Length; i++)
                test.Add(samples[indices[i]]);
        }

        private static void Shuffle(int[] indices, int? seed)
        {
            Random rnd = seed.HasValue ? new Random(seed.Value) : new Random();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rnd.Next(i + 1);
                int t = indices[i];
                indices[i] = indices[j];
                indices[j] = t;
            }
        }
    }
}
