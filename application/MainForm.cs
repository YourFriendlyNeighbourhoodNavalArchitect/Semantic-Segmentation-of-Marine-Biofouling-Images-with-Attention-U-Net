namespace BiofoulingSegmentation;

public sealed class MainForm : Form
{
    private const string MaskFolderName = "masks";
    private const int ListedErrors = 10;
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    private readonly TextBox inputFolderTextBox = new() { Width = 400, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly TextBox modelPathTextBox = new() { Width = 400, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly ProgressBar progressBar = new() { Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly Button selectInputFolderButton = new() { Text = "Select Input Folder", AutoSize = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly Button selectModelButton = new() { Text = "Select Model", AutoSize = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly Button startButton = new() { Text = "Start", AutoSize = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private CancellationTokenSource? cancellation;

    public MainForm()
    {
        Text = "Marine Biofouling Segmentation";
        AutoScaleDimensions = new SizeF(96F, 96F);
        AutoScaleMode = AutoScaleMode.Dpi;
        AutoSize = true;
        AutoSizeMode = AutoSizeMode.GrowAndShrink;
        FormBorderStyle = FormBorderStyle.FixedSingle;
        MaximizeBox = false;

        TableLayoutPanel layout = new() { ColumnCount = 2, AutoSize = true, Padding = new Padding(10) };
        layout.Controls.Add(inputFolderTextBox, 0, 0);
        layout.Controls.Add(selectInputFolderButton, 1, 0);
        layout.Controls.Add(modelPathTextBox, 0, 1);
        layout.Controls.Add(selectModelButton, 1, 1);
        layout.Controls.Add(progressBar, 0, 2);
        layout.Controls.Add(startButton, 1, 2);
        Controls.Add(layout);

        selectInputFolderButton.Click += (_, _) => SelectInputFolder();
        selectModelButton.Click += (_, _) => SelectModel();
        startButton.Click += StartButton_Click;
    }

    private void SelectInputFolder()
    {
        using FolderBrowserDialog dialog = new();
        if (dialog.ShowDialog() == DialogResult.OK)
        {
            inputFolderTextBox.Text = dialog.SelectedPath;
        }
    }

    private void SelectModel()
    {
        using OpenFileDialog dialog = new() { Filter = "ONNX models (*.onnx)|*.onnx" };
        if (dialog.ShowDialog() == DialogResult.OK)
        {
            modelPathTextBox.Text = dialog.FileName;
        }
    }

    private async void StartButton_Click(object? sender, EventArgs e)
    {
        if (cancellation is not null)
        {
            cancellation.Cancel();
            return;
        }

        string inputFolder = inputFolderTextBox.Text, modelPath = modelPathTextBox.Text;
        if (inputFolder.Length == 0 || modelPath.Length == 0)
        {
            ShowError("Select the input folder and the model.");
            return;
        }

        using CancellationTokenSource cts = cancellation = new();
        Progress<int> progress = new(value => progressBar.Value = value);
        progressBar.Value = 0;
        SetProcessing(true);
        try
        {
            var (total, errors) = await Task.Run(() => SegmentFolder(modelPath, inputFolder, progress, cts.Token));
            string summary = $"Masks saved for {total - errors.Count} of {total} images in {Path.Combine(inputFolder, MaskFolderName)}.";
            if (errors.Count > 0)
            {
                summary += "\n\nFailed:\n" + string.Join("\n", errors.Take(ListedErrors));
                if (errors.Count > ListedErrors)
                {
                    summary += $"\nand {errors.Count - ListedErrors} more.";
                }
            }
            MessageBox.Show(summary, "Segmentation complete", MessageBoxButtons.OK,
                            errors.Count == 0 ? MessageBoxIcon.Information : MessageBoxIcon.Warning);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
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
        startButton.Text = processing ? "Cancel" : "Start";
        selectInputFolderButton.Enabled = selectModelButton.Enabled = !processing;
    }

    private static void ShowError(string message) =>
        MessageBox.Show(message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);

    private static (int Total, List<string> Errors) SegmentFolder(string modelPath, string inputFolder, IProgress<int> progress,
                                                                  CancellationToken cancellationToken)
    {
        string[] imagePaths = Directory.EnumerateFiles(inputFolder).Where(path => ImageExtensions.Contains(Path.GetExtension(path))).ToArray();
        if (imagePaths.Length == 0)
        {
            throw new InvalidOperationException("Input folder contains no images.");
        }

        using Segmenter segmenter = new(modelPath);
        string maskFolder = Directory.CreateDirectory(Path.Combine(inputFolder, MaskFolderName)).FullName;
        List<string> errors = [];
        for (int i = 0; i < imagePaths.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string fileName = Path.GetFileName(imagePaths[i]);
            try
            {
                segmenter.SaveMask(imagePaths[i], Path.Combine(maskFolder, fileName + ".png"));
            }
            catch (Exception ex)
            {
                errors.Add($"{fileName}: {ex.Message}");
            }
            progress.Report((i + 1) * 100 / imagePaths.Length);
        }

        return (imagePaths.Length, errors);
    }
}
