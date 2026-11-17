import numpy as np
from hashlib import sha256
from json import load, loads, dump
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from biofouling.annotationClasses import ANNOTATION_CLASSES

SUBSETS = ('training', 'validation', 'testing')

class SubsetSplit:
    def __init__(self, metadataPath, splitPath, strategy, ratios, seed):
        self.metadataPath = metadataPath
        self.splitPath = splitPath
        self.strategy = strategy
        self.ratios = ratios
        self.seed = seed
        with open(self.metadataPath, 'r') as file:
            self.metadata = load(file)
            self.totalImages = len(self.metadata)
        self.globalClassDistribution = {i: count / self.totalImages for i, count in self.countClassIndices().items()}

    def parseSimilarity(self, similarityString):
        # Process the "Similarity" property into a set of IDs, e.g. "12-15, 18" yields {12, 13, 14, 15, 18}.
        similaritySet = set()
        for part in similarityString.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                similaritySet.update(str(ID) for ID in range(start, end + 1))
            elif part:
                similaritySet.add(str(int(part)))
        return similaritySet

    def countClassIndices(self, IDs = None):
        # Calculate class distribution.
        classIndexCount = {i: 0 for i in range(len(ANNOTATION_CLASSES))}
        relevantMetadata = self.metadata.values() if IDs is None else [self.metadata[ID] for ID in IDs]
        for metadata in relevantMetadata:
            for index in metadata.get('uniqueClassIndices', []):
                classIndexCount[index] += 1
        return classIndexCount

    def groupBySimilarity(self):
        # Group images by "Similarity" property, so that near-duplicate images never end up in different subsets.
        # Similarity is transitive: if A is similar to B, and B to C, then A, B and C form one group (union-find).
        parent = {ID: ID for ID in self.metadata}

        def find(ID):
            while parent[ID] != ID:
                parent[ID] = parent[parent[ID]]
                ID = parent[ID]
            return ID

        for ID, metadata in self.metadata.items():
            similarityString = metadata.get('similarity')
            if similarityString:
                try:
                    similarIDs = self.parseSimilarity(similarityString)
                except ValueError as error:
                    raise ValueError(f'"Similarity" property of image {ID} is malformed: "{similarityString}". '
                                     f'Expected comma-separated IDs and ranges, e.g. "12-15, 18".') from error
                # IDs that are not part of the dataset are disregarded.
                for similarID in similarIDs & parent.keys():
                    parent[find(similarID)] = find(ID)

        groups = {}
        for ID in self.metadata:
            groups.setdefault(find(ID), []).append(ID)
        return list(groups.values())

    def formGroups(self):
        # Stratified strategy: every image forms a group of its own.
        if self.strategy == 'grouped':
            return self.groupBySimilarity()
        return [[ID] for ID in self.metadata]

    def stratifiedSplit(self, groups, testSize):
        # Groups are stratified with respect to the classes they contain.
        labels = np.array([[min(count, 1) for count in self.countClassIndices(group).values()] for group in groups])
        stratifier = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = testSize, random_state = self.seed)
        firstIndices, secondIndices = next(stratifier.split(np.zeros((len(groups), 1)), labels))
        return [groups[i] for i in firstIndices], [groups[i] for i in secondIndices]

    def assignToSubsets(self):
        # Assign images to the appropriate subsets, such that:
        # 1) Similar images are placed together (grouped strategy).
        # 2) Subsets are stratified with respect to class distributions.
        # 3) Split ratios are approximately maintained.
        groups = self.formGroups()
        print(f'{self.totalImages} images form {len(groups)} groups.')
        trainingGroups, restGroups = self.stratifiedSplit(groups, self.ratios[1] + self.ratios[2])
        validationGroups, testingGroups = self.stratifiedSplit(restGroups, self.ratios[2] / (self.ratios[1] + self.ratios[2]))

        trainingSet = {ID for group in trainingGroups for ID in group}
        validationSet = {ID for group in validationGroups for ID in group}
        testingSet = {ID for group in testingGroups for ID in group}
        return trainingSet, validationSet, testingSet

    def validateSplits(self, trainingSet, validationSet, testingSet):
        totalAssigned = len(trainingSet) + len(validationSet) + len(testingSet)
        if totalAssigned != self.totalImages:
            raise ValueError(f'Mismatch in total assigned images: {totalAssigned} != {self.totalImages}.')

        # Validate that no overlap has occurred.
        if (trainingSet & validationSet) or (trainingSet & testingSet) or (validationSet & testingSet):
            raise ValueError('Overlap detected between subsets.')

        # Validate split ratios and class distribution.
        print(f'Global Class Distribution: {self.globalClassDistribution}')
        for name, subset in [('Training', trainingSet), ('Validation', validationSet), ('Testing', testingSet)]:
            distribution = {key: round(value / len(subset), 3) for key, value in self.countClassIndices(subset).items()}
            print(f'{name} Ratio: {len(subset) / self.totalImages:.3f}')
            print(f'{name} Class Distribution: {distribution}')

    def saveSplit(self, subsets):
        # The split is stored as a manifest of image IDs, rather than as copies of the files.
        self.splitPath.parent.mkdir(parents = True, exist_ok = True)
        manifest = {'strategy': self.strategy, 'seed': self.seed, 'ratios': list(self.ratios)}
        manifest.update({name: sorted(subset) for name, subset in zip(SUBSETS, subsets)})
        with open(self.splitPath, 'w') as file:
            dump(manifest, file, indent = 4)
        print(f'Split saved to {self.splitPath}.')

    def splitDataset(self):
        # Split the original dataset into training, validation and testing subsets.
        subsets = self.assignToSubsets()
        self.validateSplits(*subsets)
        self.saveSplit(subsets)

def loadSplit(splitPath):
    # Image IDs of each subset, and a short fingerprint of the manifest, to trace which split a model was trained on.
    if not splitPath.exists():
        raise FileNotFoundError(f'Split {splitPath} not found. Run "biofouling split" with the same configuration first.')
    content = splitPath.read_bytes()
    manifest = loads(content)
    return {name: manifest[name] for name in SUBSETS}, sha256(content).hexdigest()[:12]
