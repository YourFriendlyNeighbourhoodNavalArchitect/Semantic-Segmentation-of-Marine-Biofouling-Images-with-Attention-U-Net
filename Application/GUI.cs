using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace Production
{
    public partial class GUI : Form
    {
        private readonly ProgressBar progressBar;
        private readonly ToolStripStatusLabel statusLabel;
        private CancellationTokenSource? cancellationTokenSource;

        private static readonly Color ButtonDefaultColor = Color.FromArgb(192, 192, 255);
        private static readonly Color ButtonCancelColor = Color.Red;
        private const string ButtonTextStart = "Initiate Predictions";
        private const string ButtonTextCancel = "Cancel";
        private const string ErrorLogName = "errors.log";
        private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        };

        public GUI()
        {
            InitializeComponent();

            this.BackColor = Color.LightGray;
            this.Font = new Font("Segoe UI", 10);

            StatusStrip statusStrip = new();
            statusLabel = new ToolStripStatusLabel("Ready");
            statusStrip.Items.Add(statusLabel);
            this.Controls.Add(statusStrip);

            progressBar = new ProgressBar
            {
                Dock = DockStyle.Bottom,
                Maximum = 100,
                Step = 1,
                Style = ProgressBarStyle.Continuous
            };
            this.Controls.Add(progressBar);
        }

        private void SetProcessing(bool processing)
        {
            StartPredictionButton.Text = processing ? ButtonTextCancel : ButtonTextStart;
            StartPredictionButton.BackColor = processing ? ButtonCancelColor : ButtonDefaultColor;
            SelectInputFolderButton.Enabled = SelectOutputFolderButton.Enabled = SelectModelButton.Enabled = !processing;
        }

        private static void ShowError(string message)
        {
            MessageBox.Show(message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private void SelectInputFolderButton_Click(object sender, EventArgs e)
        {
            using FolderBrowserDialog fbd = new();
            if (fbd.ShowDialog() == DialogResult.OK)
            {
                InputFolderTextBox.Text = fbd.SelectedPath;
            }
        }

        private void SelectOutputFolderButton_Click(object sender, EventArgs e)
        {
            using FolderBrowserDialog fbd = new();
            if (fbd.ShowDialog() == DialogResult.OK)
            {
                OutputFolderTextBox.Text = fbd.SelectedPath;
            }
        }

        private void SelectModelButton_Click(object sender, EventArgs e)
        {
            using OpenFileDialog ofd = new()
            {
                Filter = "ONNX files (*.onnx)|*.onnx"
            };
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                ModelPathTextBox.Text = ofd.FileName;
            }
        }

        private async void StartPredictionButton_Click(object sender, EventArgs e)
        {
            if (cancellationTokenSource != null)
            {
                cancellationTokenSource.Cancel();
                return;
            }

            if (!ValidateInputs(out string inputFolder, out string outputFolder, out string modelPath))
            {
                return;
            }

            using CancellationTokenSource cts = cancellationTokenSource = new();
            // Progress<T> reports on the UI thread, where it was created.
            Progress<int> progress = new(value => progressBar.Value = value);
            progressBar.Value = 0;
            statusLabel.Text = "Processing...";
            SetProcessing(true);
            try
            {
                var (total, errors) = await Task.Run(() => PredictMasks(modelPath, inputFolder, outputFolder, progress, cts.Token));
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
                ShowError($"Processing error: {ex.Message}");
            }
            finally
            {
                cancellationTokenSource = null;
                SetProcessing(false);
            }
        }

        private static (int Total, List<string> Errors) PredictMasks(string modelPath, string inputFolder, string outputFolder,
                                                                     IProgress<int> progress, CancellationToken cancellationToken)
        {
            string[] filePaths = Directory.EnumerateFiles(inputFolder)
                                          .Where(path => ImageExtensions.Contains(Path.GetExtension(path))).ToArray();
            if (filePaths.Length == 0)
            {
                throw new InvalidOperationException("No supported images found in the input folder.");
            }

            using InferenceSession session = CreateInferenceSession(modelPath);
            // Input resolution is read from the model, of the form (B, C, H, W).
            int[] inputDimensions = session.InputMetadata[session.InputNames[0]].Dimensions;
            OpenCvSharp.Size resolution = new(inputDimensions[3], inputDimensions[2]);
            using Mat colorTable = ReadColorTable(session);

            // Images are processed one at a time, so that memory use does not grow with the folder.
            string errorLogPath = Path.Combine(outputFolder, ErrorLogName);
            File.Delete(errorLogPath);
            List<string> errors = [];
            try
            {
                for (int i = 0; i < filePaths.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    try
                    {
                        ProcessImage(session, filePaths[i], outputFolder, resolution, colorTable);
                    }
                    catch (Exception ex)
                    {
                        errors.Add($"{Path.GetFileName(filePaths[i])}: {ex.Message}");
                    }
                    progress.Report((i + 1) * 100 / filePaths.Length);
                }
            }
            finally
            {
                if (errors.Count > 0)
                {
                    File.WriteAllLines(errorLogPath, errors);
                }
            }

            return (filePaths.Length, errors);
        }

        private static InferenceSession CreateInferenceSession(string modelPath)
        {
            using SessionOptions options = new();
            options.AppendExecutionProvider_CPU();
            return new InferenceSession(modelPath, options);
        }

        private static Mat ReadColorTable(InferenceSession session)
        {
            // Classes and their RGB colours are stored in the model upon export ("biofouling export"), in output order.
            if (!session.ModelMetadata.CustomMetadataMap.TryGetValue("classes", out string? classes))
            {
                throw new InvalidOperationException("The model carries no class metadata. Export it again with \"biofouling export\".");
            }

            // Lookup table from class index to colour, in BGR order as OpenCV stores colours. Unknown indices map to black.
            using JsonDocument document = JsonDocument.Parse(classes);
            Mat colorTable = new(1, 256, MatType.CV_8UC3, Scalar.All(0));
            int classIndex = 0;
            foreach (JsonElement modelClass in document.RootElement.EnumerateArray())
            {
                JsonElement color = modelClass.GetProperty("color");
                colorTable.Set(0, classIndex++, new Vec3b((byte)color[2].GetInt32(), (byte)color[1].GetInt32(), (byte)color[0].GetInt32()));
            }

            return colorTable;
        }

        private static void ProcessImage(InferenceSession session, string filePath, string outputFolder,
                                         OpenCvSharp.Size resolution, Mat colorTable)
        {
            // Files are read and written through .NET, as OpenCV cannot open non-ASCII paths on Windows.
            // EXIF orientation is ignored, as by PIL in training.
            using Mat image = Cv2.ImDecode(File.ReadAllBytes(filePath), ImreadModes.Color | ImreadModes.IgnoreOrientation);
            if (image.Empty())
            {
                throw new InvalidDataException("The image could not be decoded.");
            }

            // Bilinear resize of the 8-bit image, BGR to RGB, scaling to [0, 1] and (1, 3, H, W) float32 layout, as in training.
            using Mat blob = CvDnn.BlobFromImage(image, 1.0 / 255, resolution, new Scalar(), swapRB: true, crop: false);
            using OrtValue input = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
                                                                      [1, 3, resolution.Height, resolution.Width], blob.Data,
                                                                      blob.Total() * sizeof(float));
            using RunOptions runOptions = new();
            using var outputs = session.Run(runOptions, session.InputNames, [input], session.OutputNames);

            // Output logits are of the form (B, C, H, W).
            long[] shape = outputs[0].GetTensorTypeAndShape().Shape;
            using Mat classIndices = ArgmaxOverClasses(outputs[0].GetTensorDataAsSpan<float>(), (int)shape[1], (int)shape[2], (int)shape[3]);
            using Mat mask = new();
            Cv2.CvtColor(classIndices, mask, ColorConversionCodes.GRAY2BGR);
            Cv2.LUT(mask, colorTable, mask);

            Cv2.ImEncode(".png", mask, out byte[] encodedMask);
            File.WriteAllBytes(Path.Combine(outputFolder, Path.GetFileName(filePath) + ".mask.png"), encodedMask);
        }

        private static Mat ArgmaxOverClasses(ReadOnlySpan<float> logits, int classCount, int height, int width)
        {
            // Each class occupies a contiguous plane, which is compared with the running maximum over the previous classes.
            int planeSize = height * width;
            float[] maxima = logits[..planeSize].ToArray();
            Mat classIndices = new(height, width, MatType.CV_8UC1, Scalar.All(0));
            Span<byte> indices = classIndices.AsSpan<byte>();
            for (int c = 1; c < classCount; c++)
            {
                ReadOnlySpan<float> plane = logits.Slice(c * planeSize, planeSize);
                for (int i = 0; i < planeSize; i++)
                {
                    if (plane[i] > maxima[i])
                    {
                        maxima[i] = plane[i];
                        indices[i] = (byte)c;
                    }
                }
            }

            return classIndices;
        }

        private bool ValidateInputs(out string inputFolder, out string outputFolder, out string modelPath)
        {
            inputFolder = InputFolderTextBox.Text;
            outputFolder = OutputFolderTextBox.Text;
            modelPath = ModelPathTextBox.Text;

            if (string.IsNullOrEmpty(inputFolder) || string.IsNullOrEmpty(outputFolder) ||
                string.IsNullOrEmpty(modelPath))
            {
                ShowError("Please provide input/output folders and a model path.");
                return false;
            }

            if (!Directory.Exists(inputFolder))
            {
                ShowError("Input folder does not exist.");
                return false;
            }

            if (!Directory.Exists(outputFolder))
            {
                try
                {
                    Directory.CreateDirectory(outputFolder);
                }
                catch (Exception ex)
                {
                    ShowError($"Could not create output folder: {ex.Message}");
                    return false;
                }
            }

            if (!File.Exists(modelPath))
            {
                ShowError("Model file does not exist.");
                return false;
            }

            return true;
        }
    }
}
