# Biofouling Segmentation (Windows application)

Predicts segmentation masks for a folder of images with a model exported by `biofouling export` (ONNX).

**Requirements:** Windows 10 or later, x64; .NET 10 SDK to build (the .NET 10 Desktop Runtime to run a framework-dependent build).

**Build and run:**

    dotnet build Application -c Release
    dotnet run --project Application -c Release

**Standalone executable** (from `Application/`; native libraries are written next to the executable, ship the whole `publish` folder):

    dotnet publish -c Release -r win-x64 --self-contained -p:PublishSingleFile=true

**Usage:** select the input folder, the output folder and the `.onnx` model, then *Initiate Predictions*.
Images (`.jpg .jpeg .png .bmp .tif .tiff .webp`, top folder only) are resized to the model input resolution.

**Output:** for each image `name.ext`, an RGB mask `name.ext.mask.png` at the model resolution, coloured with the class colours
stored in the model. Images that fail are listed in `errors.log` in the output folder.
