from logging import getLogger
from pathlib import Path
from cv2 import CAP_PROP_POS_MSEC, VideoCapture, imwrite

logger = getLogger(__name__)

def videoToFrames(videoPath: Path, outputDirectory: Path, interval: float) -> int:
    # Portion of the dataset is in the form of video footage. A frame is saved every interval seconds, so that a shorter
    # interval yields more frames. Returns the number of saved frames.
    video = VideoCapture(str(videoPath))
    if not video.isOpened():
        raise FileNotFoundError(f'Unable to open video file at {videoPath}.')
    outputDirectory.mkdir(parents = True, exist_ok = True)

    savedFrames = 0
    while True:
        video.set(CAP_PROP_POS_MSEC, savedFrames * 1000 * interval)
        success, image = video.read()
        if not success:
            break
        # Frames are named after their video, so that frames of different videos do not overwrite each other.
        imwrite(str(outputDirectory / f'{videoPath.stem} frame{savedFrames}.jpg'), image)
        logger.debug(f'Saved the frame at {savedFrames * interval:g} seconds.')
        savedFrames += 1

    video.release()
    logger.info(f'{savedFrames} frames of {videoPath.name} saved to {outputDirectory}.')
    return savedFrames
