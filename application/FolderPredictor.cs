using OpenCvSharp;

namespace BiofoulingSegmentation;

internal sealed record FolderPrediction(int Total, IReadOnlyList<string> Errors);

internal static class FolderPredictor
{
    public const string ErrorLogName = "errors.log";
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    public static FolderPrediction Run(string modelPath, string inputFolder, string outputFolder,
                                       IProgress<int> progress, CancellationToken cancellationToken)
    {
        if (IsSameFolder(inputFolder, outputFolder))
        {
            throw new ArgumentException("Output folder must differ from input folder.");
        }

        string[] imagePaths = Directory.EnumerateFiles(inputFolder).Where(path => ImageExtensions.Contains(Path.GetExtension(path))).ToArray();
        if (imagePaths.Length == 0)
        {
            throw new InvalidOperationException("Input folder contains no images.");
        }

        Directory.CreateDirectory(outputFolder);
        using Segmenter segmenter = new(modelPath);

        string errorLogPath = Path.Combine(outputFolder, ErrorLogName);
        File.Delete(errorLogPath);
        List<string> errors = [];
        try
        {
            for (int i = 0; i < imagePaths.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    SaveMask(segmenter, imagePaths[i], outputFolder);
                }
                catch (Exception ex)
                {
                    errors.Add($"{Path.GetFileName(imagePaths[i])}: {ex.Message}");
                }
                progress.Report((i + 1) * 100 / imagePaths.Length);
            }
        }
        finally
        {
            if (errors.Count > 0)
            {
                File.WriteAllLines(errorLogPath, errors);
            }
        }

        return new FolderPrediction(imagePaths.Length, errors);
    }

    private static bool IsSameFolder(string first, string second) =>
        string.Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(first)),
                      Path.TrimEndingDirectorySeparator(Path.GetFullPath(second)), StringComparison.OrdinalIgnoreCase);

    private static void SaveMask(Segmenter segmenter, string imagePath, string outputFolder)
    {
        // Files are read and written through .NET, as OpenCV cannot open non-ASCII paths on Windows.
        using Mat image = Cv2.ImDecode(File.ReadAllBytes(imagePath), ImreadModes.Color | ImreadModes.IgnoreOrientation);
        if (image.Empty())
        {
            throw new InvalidDataException("The image could not be decoded.");
        }

        using Mat mask = segmenter.Predict(image);
        Cv2.ImEncode(".png", mask, out byte[] encodedMask);
        File.WriteAllBytes(Path.Combine(outputFolder, Path.GetFileName(imagePath) + ".mask.png"), encodedMask);
    }
}
