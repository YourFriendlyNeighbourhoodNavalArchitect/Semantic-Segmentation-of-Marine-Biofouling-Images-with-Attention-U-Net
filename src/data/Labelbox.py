import numpy as np
from json import loads, dump
from requests import Session
from requests.exceptions import RequestException
from time import sleep
from os.path import splitext
from PIL import Image
from src.annotationClasses import ANNOTATION_CLASSES

# Placeholder of pixels not yet covered by any annotation, while a mask is being assembled.
UNANNOTATED = 255

class Labelbox:
    def __init__(self, inputFile, outputDirectory, metadataPath, apiKey):
        # Labelbox outputs a JSON file with all mask information.
        self.inputFile = inputFile
        self.outputDirectory = outputDirectory
        self.metadataPath = metadataPath
        # Reusable and secure connection for all HTTP requests.
        self.session = Session()
        self.session.headers.update({'Authorization': f'Bearer {apiKey}'})
        # Dictionary to store metadata for dataset stratification.
        self.metadata = {}

    def readFile(self):
        if not self.inputFile.exists():
            raise FileNotFoundError(f'File {self.inputFile} not found.')
        with open(self.inputFile, 'r') as f:
            return [loads(line) for line in f]

    def downloadMask(self, URL):
        # Individual class masks are located in respective URLs.
        maxRetries = 3
        retryDelay = 5
        requestTimeout = 30

        for attempt in range(maxRetries):
            try:
                response = self.session.get(URL, stream = True, timeout = requestTimeout)
                response.raise_for_status()
                # Each class mask is a binary file.
                mask = Image.open(response.raw).convert('L')
                return np.array(mask)
            except RequestException as e:
                print(f'Error downloading mask from {URL} (attempt {attempt + 1}/{maxRetries}): {e}')
                if attempt < maxRetries - 1:
                    sleep(retryDelay)
                else:
                    print(f'Failed to download mask from {URL} after {maxRetries} attempts.')
                    return None

    def extractSimilarity(self, classifications):
        # Extract "Similarity" property, which lists the IDs of near-duplicate images (e.g. frames of the same video).
        for classification in classifications:
            if classification.get('name') == 'Similarity':
                return classification.get('text_answer', {}).get('content')
        return None

    def processImage(self, entry):
        # Parse JSON file based on its structure.
        dataRow = entry.get('data_row', {})
        image = dataRow.get('external_id')
        ID = splitext(image)[0]
        projects = entry.get('projects', {})
        projectData = list(projects.values())[0]
        labels = projectData.get('labels', [])
        annotations = labels[0].get('annotations', {})
        objects = annotations.get('objects', [])
        similarity = self.extractSimilarity(annotations.get('classifications', []))

        fullMask = None
        for annotation in objects:
            className = annotation.get('name')
            if className not in ANNOTATION_CLASSES:
                raise KeyError(f'Class "{className}" of {ID} is not listed in ANNOTATION_CLASSES.')
            classIndex = ANNOTATION_CLASSES.index(className)
            classMask = self.downloadMask(annotation.get('mask', {}).get('url'))
            if classMask is None:
                print(f'Failed to download mask for {ID}. Skipping entry.')
                return None

            # Initialize mask with appropriate dimensions, upon the first class mask.
            if fullMask is None:
                fullMask = np.full(classMask.shape, UNANNOTATED, dtype = np.uint8)
            # Populate mask with individual class mask information, using ANNOTATION_CLASSES.
            fullMask[classMask > 0] = classIndex

        if fullMask is None:
            print(f'No annotations found for {ID}. Skipping entry.')
            return None
        # Every pixel of every image is annotated. A gap indicates a faulty export, and is reported here,
        # rather than deep into training.
        unannotatedPixels = np.count_nonzero(fullMask == UNANNOTATED)
        if unannotatedPixels:
            raise ValueError(f'{unannotatedPixels} pixels of {ID} are not covered by any annotation.')

        # Classes are read from the final mask, as overlapping annotations may conceal a class entirely.
        uniqueClasses = [int(index) for index in np.unique(fullMask)]
        self.metadata[ID] = {'uniqueClassIndices': uniqueClasses, 'similarity': similarity}
        return ID, fullMask

    def saveMetadata(self):
        # Save the metadata dictionary as a JSON file.
        with open(self.metadataPath, 'w') as f:
            dump(self.metadata, f, indent = 4)
        print(f'Metadata saved to {self.metadataPath}.')

    def saveMasks(self):
        self.outputDirectory.mkdir(parents = True, exist_ok = True)
        entries = self.readFile()
        for entry in entries:
            result = self.processImage(entry)
            if result is None:
                continue
            ID, fullMask = result

            # Save the mask locally as an NPY file.
            outputPath = self.outputDirectory / f'{ID}.npy'
            np.save(outputPath, fullMask)
            print(f'Saved mask for {ID} to {outputPath}.')

        self.saveMetadata()
