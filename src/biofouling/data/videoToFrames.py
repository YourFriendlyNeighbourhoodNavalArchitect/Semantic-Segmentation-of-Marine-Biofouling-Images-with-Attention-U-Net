from cv2 import CAP_PROP_POS_MSEC, VideoCapture, imwrite

def videoToFrames(pathIn, pathOut, frameRate):
    # Simple function that cuts given video into its frames.
    # Portion of the dataset is in the form of video footage.
    # A lower frameRate value corresponds to more snapshots.
    count = 0
    pathOut.mkdir(parents = True, exist_ok = True)
    savedFrames = 0
    video = VideoCapture(str(pathIn))
    if not video.isOpened():
        print(f'Unable to open video file at {pathIn}.')
        return

    while True:
        video.set(CAP_PROP_POS_MSEC, (count * 1000 * frameRate))
        success, image = video.read()
        if not success:
            break

        # Frames are named after their video, so that frames of different videos do not overwrite each other.
        imwrite(str(pathOut / f'{pathIn.stem} frame{count}.jpg'), image)
        print(f'Read a new frame at {count * frameRate} seconds.')
        savedFrames += 1
        count += 1

    video.release()
    print(f'Video processing complete. {savedFrames} frames saved to {pathOut}.')
