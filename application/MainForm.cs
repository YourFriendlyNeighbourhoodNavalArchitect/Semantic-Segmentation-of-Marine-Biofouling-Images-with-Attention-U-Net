namespace BiofoulingSegmentation;

public partial class MainForm : Form
{
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
            FolderPrediction result = await Task.Run(() => FolderPredictor.Run(modelPath, inputFolder, outputFolder, progress, cts.Token));
            int failed = result.Errors.Count;
            statusLabel.Text = failed == 0 ? "Completed" : $"Completed with {failed} errors";
            string summary = $"Masks saved for {result.Total - failed} of {result.Total} images.";
            if (failed > 0)
            {
                summary += $"\n{failed} images failed, see {FolderPredictor.ErrorLogName} in the output folder.";
            }
            MessageBox.Show(summary, "Predictions complete", MessageBoxButtons.OK,
                            failed == 0 ? MessageBoxIcon.Information : MessageBoxIcon.Warning);
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
}
