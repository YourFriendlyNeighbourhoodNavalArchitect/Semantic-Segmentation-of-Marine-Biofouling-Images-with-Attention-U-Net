using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Size = OpenCvSharp.Size;

namespace BiofoulingSegmentation;

/// <summary>
/// Segmentation model exported by "biofouling export", which predicts a colour mask for an image.
/// </summary>
internal sealed class Segmenter : IDisposable
{
    private readonly InferenceSession session;
    private readonly Size inputSize;
    private readonly Mat colorTable;

    public Segmenter(string modelPath)
    {
        session = new InferenceSession(modelPath);
        // Input of the form (B, C, H, W); the batch dimension is dynamic, the resolution fixed.
        int[] inputDimensions = session.InputMetadata[session.InputNames[0]].Dimensions;
        inputSize = new Size(inputDimensions[3], inputDimensions[2]);
        colorTable = ReadColorTable(session.ModelMetadata.CustomMetadataMap);
    }

    /// <summary>
    /// Predicts the class of every pixel of a BGR image, and returns the classes as a BGR colour mask at the model resolution.
    /// </summary>
    public Mat Predict(Mat image)
    {
        // Bilinear resize of the 8-bit image, BGR to RGB, scaling to [0, 1] and (1, 3, H, W) float32 layout, as in training.
        using Mat blob = CvDnn.BlobFromImage(image, 1.0 / 255, inputSize, new Scalar(), swapRB: true, crop: false);
        using OrtValue input = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
                                                                  [1, 3, inputSize.Height, inputSize.Width], blob.Data,
                                                                  blob.Total() * sizeof(float));
        using RunOptions runOptions = new();
        using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(runOptions, session.InputNames, [input], session.OutputNames);

        // Output logits of the form (B, C, H, W).
        long[] shape = outputs[0].GetTensorTypeAndShape().Shape;
        using Mat classIndices = ArgmaxOverClasses(outputs[0].GetTensorDataAsSpan<float>(), (int)shape[1], (int)shape[2], (int)shape[3]);
        Mat mask = new();
        Cv2.CvtColor(classIndices, mask, ColorConversionCodes.GRAY2BGR);
        Cv2.LUT(mask, colorTable, mask);
        return mask;
    }

    public void Dispose()
    {
        session.Dispose();
        colorTable.Dispose();
    }

    private static Mat ReadColorTable(Dictionary<string, string> metadata)
    {
        // Classes and their RGB colours are stored in the model upon export, in output order.
        if (!metadata.TryGetValue("classes", out string? classes))
        {
            throw new InvalidOperationException("The model carries no class metadata. Export it again with \"biofouling export\".");
        }

        // Lookup table from class index to colour, in BGR order as OpenCV stores colours.
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
}
