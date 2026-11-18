using OpenCvSharp;

namespace BiofoulingSegmentation;

/// <summary>
/// Window that applies a segmentation model to every image of a folder, and saves the predicted masks to another folder.
/// </summary>
public partial class MainForm : Form
{
    private const string ErrorLogName = "errors.log";
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    // The start button doubles as the cancel button while processing; its designer appearance is restored afterwards.
    private readonly string startText;
    private readonly Color startColor;
    private CancellationTokenSource? cancellation;

    public MainForm()
    {
        InitializeComponent();
        startText = startButton.Text;
        startColor = startButton.BackColor;
    }

    private void SelectInputFolderButton_Click(object sender, EventArgs e) => SelectFolder(inputFolderTextBox);

    private void SelectOutputFolderButton_Click(object sender, EventArgs e) => SelectFolder(outputFolderTextBox);

    private void SelectModelButton_Click(object sender, EventArgs e)
    {
        using OpenFileDialog dialog = new() { Filter = "ONNX models (*.onnx)|*.onnx" };
        if (dialog.ShowDialog() == DialogResult.OK)
        {
            modelPathTextBox.Text = dialog.FileName;
        }
    }

    private static void SelectFolder(TextBox target)
    {
        using FolderBrowserDialog dialog = new();
        if (dialog.ShowDialog() == DialogResult.OK)
        {
            target.Text = dialog.SelectedPath;
        }
    }

    private async void StartButton_Click(object sender, EventArgs e)
    {
        if (cancellation is not null)
        {
            cancellation.Cancel();
            return;
        }

        string inputFolder = inputFolderTextBox.Text, outputFolder = outputFolderTextBox.Text, modelPath = modelPathTextBox.Text;
        if (inputFolder.Length == 0 || outputFolder.Length == 0 || modelPath.Length == 0)
        {
            ShowError("Select the input folder, the output folder and the model.");
            return;
        }

        using CancellationTokenSource cts = cancellation = new();
        // Progress<T> reports on the UI thread, where it is created.
        Progress<int> progress = new(value => progressBar.Value = value);
        progressBar.Value = 0;
        statusLabel.Text = "Processing...";
        SetProcessing(true);
        try
        {
            var (total, errors) = await Task.Run(() => PredictFolder(modelPath, inputFolder, outputFolder, progress, cts.Token));
            statusLabel.Text = errors.Count == 0 ? "Completed" : $"Completed with {errors.Count} errors";
            string summary = $"Masks saved for {total - errors.Count} of {total} images.";
            if (errors.Count > 0)
            {
                summary += $"\n{errors.Count} images failed, see {ErrorLogName} in the output folder.";
            }
            MessageBox.Show(summary, "Predictions complete", MessageBoxButtons.OK,
                            errors.Count == 0 ? MessageBoxIcon.Information : MessageBoxIcon.Warning);
        }
        catch (OperationCanceledException)
        {
            statusLabel.Text = "Cancelled";
        }
        catch (Exception ex)
        {
            statusLabel.Text = "Failed";
            ShowError(ex.Message);
        }
        finally
        {
            cancellation = null;
            SetProcessing(false);
        }
    }

    private void SetProcessing(bool processing)
    {
        startButton.Text = processing ? "Cancel" : startText;
        startButton.BackColor = processing ? Color.Red : startColor;
        selectInputFolderButton.Enabled = selectOutputFolderButton.Enabled = selectModelButton.Enabled = !processing;
    }

    private static void ShowError(string message) =>
        MessageBox.Show(message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);

    private static (int Total, List<string> Errors) PredictFolder(string modelPath, string inputFolder, string outputFolder,
                                                                  IProgress<int> progress, CancellationToken cancellationToken)
    {
        // Images are processed one at a time, so that memory use does not grow with the folder.
        string[] imagePaths = Directory.EnumerateFiles(inputFolder).Where(path => ImageExtensions.Contains(Path.GetExtension(path))).ToArray();
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
            // Errors are logged even if processing is cancelled.
            if (errors.Count > 0)
            {
                File.WriteAllLines(errorLogPath, errors);
            }
        }

        return (imagePaths.Length, errors);
    }

    private static void SaveMask(Segmenter segmenter, string imagePath, string outputFolder)
    {
        // Files are read and written through .NET, as OpenCV cannot open non-ASCII paths on Windows.
        // EXIF orientation is ignored, as by PIL in training.
        using Mat image = Cv2.ImDecode(File.ReadAllBytes(imagePath), ImreadModes.Color | ImreadModes.IgnoreOrientation);
        if (image.Empty())
        {
            throw new InvalidDataException("The image could not be decoded.");
        }

        using Mat mask = segmenter.Predict(image);
        Cv2.ImEncode(".png", mask, out byte[] encodedMask);
        // The extension is kept in the name, so that images that differ only in their extension do not overwrite each other.
        File.WriteAllBytes(Path.Combine(outputFolder, Path.GetFileName(imagePath) + ".mask.png"), encodedMask);
    }
}
