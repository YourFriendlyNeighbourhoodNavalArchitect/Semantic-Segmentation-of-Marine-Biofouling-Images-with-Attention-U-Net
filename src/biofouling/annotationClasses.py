# Images are annotated in Labelbox with the classes below. Stored masks hold the position of each class in this tuple,
# so the order must never change.
ANNOTATION_CLASSES = ('Clean Hull', 'Slime/Algae', 'Calcareous Deposits', 'Barnacles/Molluscs', 'Background/Other')

def classRemap(classes):
    # Lookup table from annotation index to model index, for classes given as dictionaries with "name" and "annotations".
    remap = [None] * len(ANNOTATION_CLASSES)
    for modelIndex, modelClass in enumerate(classes):
        for annotationClass in modelClass['annotations']:
            if annotationClass not in ANNOTATION_CLASSES:
                raise ValueError(f'Class "{modelClass["name"]}" absorbs "{annotationClass}", which is not an annotation class. '
                                 f'Annotation classes: {", ".join(ANNOTATION_CLASSES)}.')
            annotationIndex = ANNOTATION_CLASSES.index(annotationClass)
            if remap[annotationIndex] is not None:
                raise ValueError(f'Annotation class "{annotationClass}" is absorbed by more than one class.')
            remap[annotationIndex] = modelIndex
    missing = [name for name, modelIndex in zip(ANNOTATION_CLASSES, remap) if modelIndex is None]
    if missing:
        raise ValueError(f'Annotation classes not absorbed by any class: {", ".join(missing)}.')
    return tuple(remap)
