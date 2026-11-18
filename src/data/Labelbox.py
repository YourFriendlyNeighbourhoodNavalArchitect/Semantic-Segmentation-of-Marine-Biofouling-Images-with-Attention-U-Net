import json
import numpy as np
from io import BytesIO
from logging import getLogger
from pathlib import Path
from time import sleep
from typing import Any
from PIL import Image
from requests import Session
from requests.exceptions import RequestException
from src.annotationClasses import ANNOTATION_CLASSES

logger = getLogger(__name__)

# Placeholder of pixels not yet covered by any annotation, while a mask is being assembled.
UNANNOTATED = 255
MAX_ATTEMPTS = 3
RETRY_DELAY = 5
REQUEST_TIMEOUT = 30

def requireField(container: Any, key: str, expectedType: type, context: str) -> Any:
    # Fields of the export are checked one by one, so that an unexpected structure is reported with its location.
    value = container.get(key) if isinstance(container, dict) else None
    if not isinstance(value, expectedType):
        raise ValueError(f'{context}: field "{key}" is missing, or is not of type {expectedType.__name__}.')
    return value

class Labelbox:
    def __init__(self, inputFile: Path, outputDirectory: Path, metadataPath: Path, apiKey: str | None):
        # Masks are downloaded with the API key, so a missing key is reported before any request is made.
        if not apiKey:
            raise ValueError('The Labelbox API key is missing. Set LABELBOX_API_KEY in .env (see .env.example).')
        # Labelbox outputs a JSON file with all mask information.
        self.inputFile = inputFile
        self.outputDirectory = outputDirectory
        self.metadataPath = metadataPath
        # Reusable connection for all HTTP requests, closed once every mask is saved.
        self.session = Session()
        self.session.headers.update({'Authorization': f'Bearer {apiKey}'})
        # Dictionary to store metadata for dataset stratification.
        self.metadata = {}

    def readFile(self) -> list[tuple[int, dict[str, Any]]]:
        # Entries are numbered by their line, to locate them in error messages.
        if not self.inputFile.exists():
            raise FileNotFoundError(f'File {self.inputFile} not found.')
        with open(self.inputFile, 'r', encoding = 'utf-8') as file:
            return [(lineNumber, json.loads(line)) for lineNumber, line in enumerate(file, start = 1) if line.strip()]

    def downloadMask(self, URL: str) -> np.ndarray | None:
        # Individual class masks are binary images, located in respective URLs.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                # The response is closed once read, so that its connection returns to the session.
                with self.session.get(URL, timeout = REQUEST_TIMEOUT) as response:
                    response.raise_for_status()
                    content = response.content
                return np.array(Image.open(BytesIO(content)).convert('L'))
            # Network errors, and images that cannot be decoded (PIL raises OSError, e.g. for a truncated download).
            except (RequestException, OSError) as error:
                logger.warning(f'Mask {URL} could not be downloaded or decoded (attempt {attempt}/{MAX_ATTEMPTS}): {error}')
                if attempt < MAX_ATTEMPTS:
                    sleep(RETRY_DELAY)
        return None

    def extractSimilarity(self, classifications: list[dict[str, Any]]) -> str | None:
        # Extract "Similarity" property, which lists the IDs of near-duplicate images (e.g. frames of the same video).
        for classification in classifications:
            if isinstance(classification, dict) and classification.get('name') == 'Similarity':
                return (classification.get('text_answer') or {}).get('content')
        return None

    def parseEntry(self, entry: dict[str, Any], lineNumber: int) -> tuple[str, list[dict[str, Any]], str | None]:
        # Structure of an entry: data_row.external_id, and projects.<project>.labels[0].annotations.
        context = f'Line {lineNumber} of {self.inputFile.name}'
        image = requireField(requireField(entry, 'data_row', dict, context), 'external_id', str, context)
        ID = Path(image).stem
        context = f'Image {ID} (line {lineNumber} of {self.inputFile.name})'
        projects = requireField(entry, 'projects', dict, context)
        if len(projects) != 1:
            raise ValueError(f'{context}: expected labels from one project, found {len(projects)}.')
        labels = requireField(next(iter(projects.values())), 'labels', list, context)
        if not labels:
            raise ValueError(f'{context}: no label.')
        if len(labels) > 1:
            logger.warning(f'{context}: {len(labels)} labels; the first one is used.')
        annotations = requireField(labels[0], 'annotations', dict, context)
        objects = requireField(annotations, 'objects', list, context)
        similarity = self.extractSimilarity(annotations.get('classifications') or [])
        return ID, objects, similarity

    def processImage(self, entry: dict[str, Any], lineNumber: int) -> tuple[str, np.ndarray] | None:
        ID, objects, similarity = self.parseEntry(entry, lineNumber)

        fullMask = None
        for annotation in objects:
            className = requireField(annotation, 'name', str, f'Image {ID}')
            if className not in ANNOTATION_CLASSES:
                raise ValueError(f'Class "{className}" of image {ID} is not an annotation class. '
                                 f'Annotation classes: {", ".join(ANNOTATION_CLASSES)}.')
            URL = requireField(requireField(annotation, 'mask', dict, f'Image {ID}, class "{className}"'), 'url', str,
                               f'Image {ID}, class "{className}"')
            classMask = self.downloadMask(URL)
            if classMask is None:
                logger.warning(f'Mask of class "{className}" of image {ID} could not be obtained. Image skipped.')
                return None

            # Initialize mask with appropriate dimensions, upon the first class mask.
            if fullMask is None:
                fullMask = np.full(classMask.shape, UNANNOTATED, dtype = np.uint8)
            elif classMask.shape != fullMask.shape:
                raise ValueError(f'Mask of class "{className}" of image {ID} is {classMask.shape}, unlike the others ({fullMask.shape}).')
            # Populate mask with individual class mask information, using ANNOTATION_CLASSES.
            fullMask[classMask > 0] = ANNOTATION_CLASSES.index(className)

        if fullMask is None:
            logger.warning(f'No annotations found for image {ID}. Image skipped.')
            return None
        # Every pixel of every image is annotated. A gap indicates a faulty export, and is reported here,
        # rather than deep into training.
        unannotatedPixels = np.count_nonzero(fullMask == UNANNOTATED)
        if unannotatedPixels:
            raise ValueError(f'{unannotatedPixels} pixels of image {ID} are not covered by any annotation.')

        # Classes are read from the final mask, as overlapping annotations may conceal a class entirely.
        uniqueClasses = [int(index) for index in np.unique(fullMask)]
        self.metadata[ID] = {'uniqueClassIndices': uniqueClasses, 'similarity': similarity}
        return ID, fullMask

    def saveMetadata(self) -> None:
        with open(self.metadataPath, 'w', encoding = 'utf-8') as file:
            json.dump(self.metadata, file, indent = 4)
        logger.info(f'Metadata of {len(self.metadata)} images saved to {self.metadataPath}.')

    def saveMasks(self) -> None:
        self.outputDirectory.mkdir(parents = True, exist_ok = True)
        entries = self.readFile()
        with self.session:
            for lineNumber, entry in entries:
                result = self.processImage(entry, lineNumber)
                if result is None:
                    continue
                ID, fullMask = result
                outputPath = self.outputDirectory / f'{ID}.npy'
                np.save(outputPath, fullMask)
                logger.info(f'Saved mask of image {ID} to {outputPath}.')
        self.saveMetadata()
