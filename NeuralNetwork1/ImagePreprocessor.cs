    using System;
using System.Drawing;
using System.Drawing.Imaging;
using AForge.Imaging;
using AForge.Imaging.Filters;

namespace AForge.WindowsForms
{
    static class ImagePreprocessor
    {
        public static double[] BitmapToVector100(Bitmap bitmap, Settings settings, bool skipGrayscale = false)
        {
            using (Bitmap prepared = PreprocessToBitmap100(bitmap, settings, skipGrayscale))
            {
                int w = prepared.Width;
                int h = prepared.Height;

                double[] v = new double[w * h];

                BitmapData data = prepared.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                try
                {
                    unsafe
                    {
                        byte* ptr = (byte*)data.Scan0;
                        int stride = data.Stride;

                        for (int y = 0; y < h; y++)
                        {
                            byte* row = ptr + y * stride;
                            int baseIndex = y * w;
                            for (int x = 0; x < w; x++)
                                v[baseIndex + x] = row[x] > 0 ? 1.0 : 0.0;
                        }
                    }
                }
                finally
                {
                    prepared.UnlockBits(data);
                }

                return v;
            }
        }

        // Добавлен флаг skipCrop (по умолчанию false — поведение прежнее).
        public static Bitmap PreprocessToBitmap100(Bitmap bitmap, Settings settings, bool skipGrayscale = false, bool skipCrop = false)
        {
            Bitmap roi;
            if (!skipCrop)
            {
                // Обычное кадрирование "как вебкам"
                roi = CropLikeWebcam(bitmap, settings);
            }
            else
            {
                // Не кадрируем — делаем копию в 24bpp, чтобы дальнейшие фильтры работали корректно
                roi = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                using (Graphics g = Graphics.FromImage(roi))
                    g.DrawImage(bitmap, new Rectangle(0, 0, roi.Width, roi.Height),
                        new Rectangle(0, 0, bitmap.Width, bitmap.Height), GraphicsUnit.Pixel);
            }

            UnmanagedImage u = AForge.Imaging.UnmanagedImage.FromManagedImage(roi);
            roi.Dispose();

            // Если необходимо — конвертируем в 8bpp grayscale.
            // При skipGrayscale мы пропускаем явное преобразование только если источник уже 8bpp.
            if (!skipGrayscale)
            {
                Grayscale gray = new Grayscale(0.2125, 0.7154, 0.0721);
                u = gray.Apply(u);
            }
            else
            {
                if (u.PixelFormat != PixelFormat.Format8bppIndexed)
                {
                    // источник не 8bpp — всё равно конвертируем, иначе Bradley не сработает
                    Grayscale gray = new Grayscale(0.2125, 0.7154, 0.0721);
                    u = gray.Apply(u);
                }
            }

            ResizeBilinear resize500 = new ResizeBilinear(500, 500);
            u = resize500.Apply(u);

            BradleyLocalThresholding bradley = new BradleyLocalThresholding();
            bradley.PixelBrightnessDifferenceLimit = settings != null ? settings.differenceLim : 0.15f;
            bradley.ApplyInPlace(u);

            // После порога приводим фон к белому — как в Python (если среднее слишком тёмное, инвертируем)
            bool needInvert = false;
            Bitmap tmpForMean = u.ToManagedImage();
            try
            {
                BitmapData bd = tmpForMean.LockBits(new Rectangle(0, 0, tmpForMean.Width, tmpForMean.Height), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                try
                {
                    unsafe
                    {
                        byte* ptr = (byte*)bd.Scan0;
                        long sum = 0;
                        int stride = bd.Stride;
                        int h = bd.Height;
                        int w = bd.Width;

                        for (int y = 0; y < h; y++)
                        {
                            byte* row = ptr + y * stride;
                            for (int x = 0; x < w; x++)
                                sum += row[x];
                        }

                        double mean = sum / (double)(bd.Height * bd.Width);
                        if (mean < 127.0) needInvert = true;
                    }
                }
                finally
                {
                    tmpForMean.UnlockBits(bd);
                }
            }
            finally
            {
                tmpForMean.Dispose();
            }

            if (needInvert)
            {
                Invert inv = new Invert();
                inv.ApplyInPlace(u);
            }

            // Обрезаем по содержимому — аналог coords min/max из Python
            if (!skipCrop)
            {
                u = CropToContent(u);
            }

            // Теперь масштабируем с сохранением соотношения и центрируем на белом canvas 100x100 (или 128x128 при необходимости)
            int target = 128;
            int srcW = u.Width;
            int srcH = u.Height;

            if (srcW <= 0 || srcH <= 0)
            {
                // На всякий случай: если пусто — вернуть белый холст
                Bitmap empty = new Bitmap(target, target, PixelFormat.Format24bppRgb);
                using (Graphics g = Graphics.FromImage(empty))
                {
                    g.Clear(Color.White);
                }
                return Ensure8bpp(empty);
            }

            double scale = (double)target / System.Math.Max(srcW, srcH);
            int newW = System.Math.Max(1, (int)System.Math.Round(srcW * scale));
            int newH = System.Math.Max(1, (int)System.Math.Round(srcH * scale));

            ResizeBilinear resizeSmall = new ResizeBilinear(newW, newH);
            UnmanagedImage smallUn = resizeSmall.Apply(u);

            Bitmap smallBmp = smallUn.ToManagedImage();
            smallUn.Dispose();
            u.Dispose();

            // Создаём белый canvas (24bpp) и рисуем на нём маленькое изображение по центру
            Bitmap canvas = new Bitmap(target, target, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(canvas))
            {
                g.Clear(Color.White);
                int x = (target - smallBmp.Width) / 2;
                int y = (target - smallBmp.Height) / 2;
                g.DrawImage(smallBmp, x, y, smallBmp.Width, smallBmp.Height);
            }

            smallBmp.Dispose();

            // Возвращаем 8bpp результат
            return Ensure8bpp(canvas);
        }

        private static Bitmap CropLikeWebcam(Bitmap bitmap, Settings settings)
        {
            int w = bitmap.Width;
            int h = bitmap.Height;

            int side = System.Math.Min(w, h);
            int x0 = (w - side) / 2;
            int y0 = (h - side) / 2;

            int border = settings != null ? settings.border : 0;
            int top = settings != null ? settings.top : 0;
            int left = settings != null ? settings.left : 0;

            if (side < 4 * border) border = side / 4;

            int inner = side - 2 * border;
            if (inner <= 1) inner = side;

            int cx = x0 + border + left;
            int cy = y0 + border + top;

            if (cx < 0) cx = 0;
            if (cy < 0) cy = 0;
            if (cx + inner > w) cx = w - inner;
            if (cy + inner > h) cy = h - inner;

            Rectangle rect = new Rectangle(cx, cy, inner, inner);

            Bitmap dst = new Bitmap(rect.Width, rect.Height, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
                g.DrawImage(bitmap, new Rectangle(0, 0, dst.Width, dst.Height), rect, GraphicsUnit.Pixel);

            return dst;
        }

        private static UnmanagedImage CropToContent(UnmanagedImage u)
        {
            BlobCounter bc = new BlobCounter
            {
                FilterBlobs = true,
                MinWidth = 3,
                MinHeight = 3,
                ObjectsOrder = ObjectsOrder.Size
            };

            bc.ProcessImage(u);
            Rectangle[] rects = bc.GetObjectsRectangles();
            if (rects == null || rects.Length == 0) return u;

            int lx = u.Width;
            int ly = u.Height;
            int rx = 0;
            int ry = 0;

            for (int i = 0; i < rects.Length; i++)
            {
                Rectangle r = rects[i];
                if (r.X < lx) lx = r.X;
                if (r.Y < ly) ly = r.Y;
                if (r.Right > rx) rx = r.Right;
                if (r.Bottom > ry) ry = r.Bottom;
            }

            if (rx <= lx || ry <= ly) return u;

            Crop crop = new Crop(new Rectangle(lx, ly, rx - lx, ry - ly));
            return crop.Apply(u);
        }

        private static Bitmap Ensure8bpp(Bitmap bmp)
        {
            if (bmp.PixelFormat == PixelFormat.Format8bppIndexed) return bmp;

            Grayscale gray = new Grayscale(0.2125, 0.7154, 0.0721);
            Bitmap b2 = gray.Apply(bmp);
            bmp.Dispose();
            return b2;
        }
    }
}
