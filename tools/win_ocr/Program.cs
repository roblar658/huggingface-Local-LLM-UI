using System;
using System.IO;
using System.Linq;
using System.Collections.Specialized;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Windows.Globalization;
using Windows.Graphics.Imaging;
using Windows.Media.Ocr;
using Windows.Storage;

namespace WinOcr;

class Program
{
    static int Main(string[] args)
    {
        return MainAsync(args).GetAwaiter().GetResult();
    }

    static async Task<int> MainAsync(string[] args)
    {
        try
        {
            string? filePath = null;
            if (args.Length > 0 && File.Exists(args[0]))
            {
                filePath = Path.GetFullPath(args[0]);
            }

            string? tempFile = null;

            if (filePath == null)
            {
                // Sjekk utklippstavlen på en eksplisitt STA-tråd
                Image? clipImage = null;
                StringCollection? fileDropList = null;

                var sta = new Thread(() =>
                {
                    try
                    {
                        if (Clipboard.ContainsImage())
                        {
                            clipImage = Clipboard.GetImage();
                        }
                        else if (Clipboard.ContainsFileDropList())
                        {
                            fileDropList = Clipboard.GetFileDropList();
                        }
                    }
                    catch { }
                });
                sta.SetApartmentState(ApartmentState.STA);
                sta.Start();
                sta.Join();

                if (clipImage != null)
                {
                    tempFile = Path.Combine(Path.GetTempPath(), $"ocr_clip_{Guid.NewGuid():N}.png");
                    clipImage.Save(tempFile, ImageFormat.Png);
                    clipImage.Dispose();
                    filePath = tempFile;
                }
                else if (fileDropList != null)
                {
                    foreach (var f in fileDropList)
                    {
                        if (f != null && File.Exists(f) && IsImageFile(f))
                        {
                            filePath = f;
                            break;
                        }
                    }
                }
            }

            if (filePath == null || !File.Exists(filePath))
            {
                Console.Error.WriteLine("INGEN_BILDE: Fant verken bildefil eller bilde i utklippstavlen.");
                return 2;
            }

            StorageFile storageFile = await StorageFile.GetFileFromPathAsync(filePath);
            using var stream = await storageFile.OpenAsync(FileAccessMode.Read);
            var decoder = await BitmapDecoder.CreateAsync(stream);
            var softwareBitmap = await decoder.GetSoftwareBitmapAsync();

            OcrEngine? engine = OcrEngine.TryCreateFromUserProfileLanguages();
            if (engine == null)
            {
                engine = OcrEngine.TryCreateFromLanguage(new Language("nb"));
            }
            if (engine == null)
            {
                engine = OcrEngine.TryCreateFromLanguage(new Language("en-US"));
            }
            if (engine == null)
            {
                var supported = OcrEngine.AvailableRecognizerLanguages;
                if (supported.Count > 0)
                {
                    engine = OcrEngine.TryCreateFromLanguage(supported[0]);
                }
            }

            if (engine == null)
            {
                Console.Error.WriteLine("FEIL: Ingen OCR-språkmotor er installert på maskinen.");
                return 3;
            }

            var ocrResult = await engine.RecognizeAsync(softwareBitmap);

            if (tempFile != null)
            {
                try { File.Delete(tempFile); } catch { }
            }

            var lines = ocrResult.Lines;
            var sb = new System.Text.StringBuilder();

            if (lines != null && lines.Count > 0)
            {
                // Beregn gjennomsnittlig linjehøyde for å skille vanlige linjeskift fra avsnitt
                double totalHeight = 0;
                int heightCount = 0;
                foreach (var l in lines)
                {
                    if (l.Words != null && l.Words.Count > 0)
                    {
                        totalHeight += l.Words.Average(w => w.BoundingRect.Height);
                        heightCount++;
                    }
                }
                double avgLineHeight = heightCount > 0 ? (totalHeight / heightCount) : 18.0;

                double lastBottom = -1.0;

                for (int i = 0; i < lines.Count; i++)
                {
                    var line = lines[i];
                    string lineText = line.Text?.Trim() ?? "";
                    if (string.IsNullOrEmpty(lineText)) continue;

                    if (line.Words != null && line.Words.Count > 0)
                    {
                        double curTop = line.Words.Min(w => w.BoundingRect.Y);
                        double curBottom = line.Words.Max(w => w.BoundingRect.Y + w.BoundingRect.Height);

                        if (lastBottom >= 0)
                        {
                            double gap = curTop - lastBottom;
                            // Dersom det er et tydelig vertikalt gap (avsnitt i bildet), legg til et ekstra linjeskift
                            if (gap > avgLineHeight * 0.6)
                            {
                                sb.AppendLine();
                            }
                        }
                        lastBottom = curBottom;
                    }

                    // Hver linje i bildet legges inn med sitt eget linjeskift
                    sb.AppendLine(lineText);
                }
            }

            string text = sb.Length > 0 ? sb.ToString().TrimEnd() : (ocrResult.Text ?? "").Trim();

            if (string.IsNullOrWhiteSpace(text))
            {
                Console.Error.WriteLine("TOM_TEKST: Bildet ble analysert, men ingen tekst ble funnet.");
                return 4;
            }

            // Skriv ut gjenkjent tekst med bevarte linjeskift
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine(text);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("OCR_FEIL: " + ex.Message);
            return 1;
        }
    }

    static bool IsImageFile(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext is ".png" or ".jpg" or ".jpeg" or ".bmp" or ".webp" or ".gif" or ".tiff";
    }
}
