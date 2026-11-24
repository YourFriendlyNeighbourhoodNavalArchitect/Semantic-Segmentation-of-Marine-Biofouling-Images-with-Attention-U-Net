namespace BiofoulingSegmentation;

public sealed class MainForm : Form
{
    private const string MaskFolderName = "masks";
    private const int ListedFailures = 10;
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    private readonly TextBox inputFolderTextBox = new() { Width = 400, ReadOnly = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly TextBox modelPathTextBox = new() { Width = 400, ReadOnly = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly Button selectInputFolderButton = new() { Text = "Select Input Folder", AutoSize = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly Button selectModelButton = new() { Text = "Select Model", AutoSize = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
    private readonly Button startButton = new() { Text = "Start", AutoSize = true, Anchor = AnchorStyles.None, Enabled = false };
    private readonly ProgressBar progressBar = new() { Height = 6, Anchor = AnchorStyles.Left | AnchorStyles.Right };
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
        layout.Controls.Add(startButton, 0, 2);
        layout.SetColumnSpan(startButton, 2);
        layout.Controls.Add(progressBar, 0, 3);
        layout.SetColumnSpan(progressBar, 2);
        Controls.Add(layout);

        selectInputFolderButton.Click += (_, _) => SelectInputFolder();
        selectModelButton.Click += (_, _) => SelectModel();
        startButton.Click += StartButton_Click;
        EventHandler updateStartButton = (_, _) => startButton.Enabled = inputFolderTextBox.TextLength > 0 && modelPathTextBox.TextLength > 0;
        inputFolderTextBox.TextChanged += updateStartButton;
        modelPathTextBox.TextChanged += updateStartButton;
        FormClosing += (_, e) => e.Cancel = cancellation is not null && e.CloseReason == CloseReason.UserClosing;
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
        using CancellationTokenSource cts = cancellation = new();
        Progress<int> progress = new(value => progressBar.Value = value);
        SetProcessing(true);
        (string Text, string Caption, MessageBoxIcon Icon) message;
        try
        {
            var (total, segmented, failed) = await Task.Run(() => SegmentFolder(modelPath, inputFolder, progress, cts.Token));
            message = Summarize(total, segmented, failed);
        }
        catch (Exception ex)
        {
            message = (ex is SegmentationException ? ex.Message : "Failed to segment the photos.", "Segmentation failed", MessageBoxIcon.Error);
        }

        cancellation = null;
        SetProcessing(false);
        MessageBox.Show(this, message.Text, message.Caption, MessageBoxButtons.OK, message.Icon);
        progressBar.Value = 0;
    }

    private void SetProcessing(bool processing)
    {
        startButton.Text = processing ? "Cancel" : "Start";
        selectInputFolderButton.Enabled = selectModelButton.Enabled = !processing;
    }

    private static (string Text, string Caption, MessageBoxIcon Icon) Summarize(int total, int segmented, List<string> failed)
    {
        bool cancelled = segmented + failed.Count < total;
        string text = $"Segmented {segmented}/{total} photos{(cancelled ? " before cancelling" : "")}.";
        if (failed.Count > 0)
        {
            text += $"\n\nFailed to segment {string.Join(", ", failed.Take(ListedFailures))}"
                  + (failed.Count > ListedFailures ? $" and {failed.Count - ListedFailures} more." : ".");
        }

        return (text, cancelled ? "Segmentation cancelled" : "Segmentation complete",
                failed.Count == 0 ? MessageBoxIcon.Information : MessageBoxIcon.Warning);
    }

    private static (int Total, int Segmented, List<string> Failed) SegmentFolder(string modelPath, string inputFolder, IProgress<int> progress,
                                                                               CancellationToken cancellationToken)
    {
        string[] photoPaths = Directory.EnumerateFiles(inputFolder).Where(path => ImageExtensions.Contains(Path.GetExtension(path))).ToArray();
        if (photoPaths.Length == 0)
        {
            throw new SegmentationException("Failed to find any photos in the input folder.");
        }

        string maskFolder = Path.Combine(inputFolder, MaskFolderName);
        try
        {
            Directory.CreateDirectory(maskFolder);
        }
        catch (Exception)
        {
            throw new SegmentationException("Failed to access the input folder.");
        }

        using Segmenter segmenter = new(modelPath);
        int segmented = 0;
        List<string> failed = [];
        for (int i = 0; i < photoPaths.Length && !cancellationToken.IsCancellationRequested; i++)
        {
            string fileName = Path.GetFileName(photoPaths[i]);
            try
            {
                segmenter.SaveMask(photoPaths[i], Path.Combine(maskFolder, fileName + ".png"));
                segmented++;
            }
            catch (Exception)
            {
                failed.Add(fileName);
            }
            progress.Report((i + 1) * 100 / photoPaths.Length);
        }

        return (photoPaths.Length, segmented, failed);
    }
}
