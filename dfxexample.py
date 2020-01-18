# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

import libdfx as dfx

# This function saves chunks to disk. Normally you would make a API call to the
# DFX Server instead.
def savePayload(chunkPayload, output):
    props = {
        "valid": chunkPayload.valid,
        "start_frame": chunkPayload.start_frame,
        "end_frame": chunkPayload.end_frame,
        "chunk_number": chunkPayload.chunk_number,
        "number_chunks": chunkPayload.number_chunks,
        "first_chunk_start_time_s": chunkPayload.first_chunk_start_time_s,
        "start_time_s": chunkPayload.start_time_s,
        "end_time_s": chunkPayload.end_time_s,
        "duration_s": chunkPayload.duration_s,
    }
    with open(
            os.path.join(output, "properties{}.json".format(
                chunkPayload.chunk_number)), "w") as f:
        json.dump(props, f)
    with open(
            os.path.join(output,
                         "payload{}.bin".format(chunkPayload.chunk_number)),
            "wb") as f:
        f.write(chunkPayload.payload_data)
    with open(
            os.path.join(output,
                         "metadata{}.bin".format(chunkPayload.chunk_number)),
            "wb") as f:
        f.write(chunkPayload.metadata)


def doExtraction(videoPath, facePath, studyPath, output):
    # Create a DFX Factory object
    factory = dfx.Factory()
    print("Created DFX Factory:", factory.getVersion())

    # Initialize a study
    if not factory.initializeStudyFromFile(studyPath):
        print("DFX study initialization failed: {}".format(
            factory.getLastErrorMessage()))
        sys.exit(1)
    print("Created study from {}".format(studyPath))

    # Create collector
    collector = factory.createCollector()
    if collector.getCollectorState() == dfx.CollectorState.ERROR:
        print("Collector creation failed: {}".format(
            collector.getLastErrorMessage()))
        sys.exit(1)
    print("Created collector")

    # Load the face tracking data
    with open(facePath, 'r') as f:
        videofaces = json.load(f)["frames"]

    # This function loads previously saved face tracking data.
    # Normally, you would run a face tracker on the image
    def createDFXFace(jsonFace):
        face = collector.createFace(jsonFace["id"])
        face.setRect(jsonFace['rect.x'], jsonFace['rect.y'],
                     jsonFace['rect.w'], jsonFace['rect.h'])
        face.setPoseValid(jsonFace['poseValid'])
        face.setDetected(jsonFace['detected'])
        points = jsonFace['points']
        for pointId, point in points.items():
            face.addPosePoint(pointId,
                              point['x'],
                              point['y'],
                              valid=point['valid'],
                              estimated=point['estimated'],
                              quality=point['quality'])
        return face

    # Load video file
    videocap = cv2.VideoCapture(videoPath)
    targetFPS = videocap.get(cv2.CAP_PROP_FPS)
    durationOneFrame_ns = 1000000000.0 / targetFPS

    # Set target FPS and chunk duration
    chunkDuration_s = 5
    videoDuration_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoDuration_s = videoDuration_frames / targetFPS
    numberChunks = math.ceil(videoDuration_s /
                             chunkDuration_s)  # Ask more chunks then needed

    collector.setTargetFPS(targetFPS)
    collector.setChunkDurationSeconds(chunkDuration_s)
    collector.setNumberChunks(numberChunks)

    print("    mode: {}".format(factory.getMode()))
    print("    number chunks: {}".format(collector.getNumberChunks()))
    print("    chunk duration: {}s".format(
        collector.getChunkDurationSeconds()))
    for constraint in collector.getEnabledConstraints():
        print("    enabled constraint: {}".format(constraint))

    # Create output folder if it doesn't exist
    videoFileName = os.path.basename(videoPath)
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)

    # Start collection
    collector.startCollection()

    # Start reading frames and adding to collector
    frameNumber = 0
    success = False
    while (True):
        ret, image = videocap.read()

        if image is None:
            # Video ended, so grab what should be the last, possibly truncated chunk
            chunkData = collector.getChunkData()
            if chunkData is not None:
                chunkPayload = chunkData.getChunkPayload()
                if output is not None:
                    savePayload(chunkPayload, output)
                print("Got chunk with {}".format(chunkPayload))
            else:
                print("Got empty chunk")
            success = True
            break

        # Create a dfx_video_frame
        frameNumber = int(videocap.get(cv2.CAP_PROP_POS_FRAMES))
        videoFrame = dfx.VideoFrame(image, frameNumber,
                                    frameNumber * durationOneFrame_ns,
                                    dfx.ChannelOrder.CHANNEL_ORDER_BGR)

        # Create a dfx_frame from the dfx_video_frame
        frame = collector.createFrame(videoFrame)

        # Add the dfx_face to the dfx_frame
        face = createDFXFace(videofaces[str(frameNumber)])
        frame.addFace(face)

        # Add a marker to the 1000th dfx_frame
        if frameNumber == 1000:
            frame.addMarker("This is the 1000th frame")

        # Do the extraction
        collector.defineRegions(frame)
        result = collector.extractChannels(frame)

        # Grab a chunk and check if we are finished
        if result == dfx.CollectorState.CHUNKREADY or result == dfx.CollectorState.COMPLETED:
            chunkData = collector.getChunkData()
            if chunkData is not None:
                chunkPayload = chunkData.getChunkPayload()
                if output is not None:
                    savePayload(chunkPayload, output)
                print("Got chunk with {}".format(chunkPayload))
            else:
                print("Got empty chunk")
            if result == dfx.CollectorState.COMPLETED:
                print("dfx.CollectorState.COMPLETED at frame {}".format(
                    frameNumber))
                success = True
                break

        # Render every 10th frame
        if frameNumber % 10 == 0:
            for faceID in frame.getFaceIdentifiers():
                for regionID in frame.getRegionNames(faceID):
                    if (frame.getRegionIntProperty(faceID, regionID, "draw") !=
                            0):
                        polygon = frame.getRegionPolygon(faceID, regionID)
                        cv2.polylines(image, [np.array(polygon)],
                                      isClosed=True,
                                      color=(255, 255, 0),
                                      thickness=1,
                                      lineType=cv2.LINE_AA)

            msg = "Extracting from {} - frame {} of {}".format(
                videoFileName, frameNumber, videoDuration_frames)
            cv2.putText(image,
                        msg,
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                success = False
                break

    if success:
        print("Collection finished completely")
    else:
        print("Collection interrupted or failed")

    # When everything done, release the capture
    videocap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test analyzer or run on saved payloads")
    parser.add_argument("-v",
                        "--version",
                        action='version',
                        version='%(prog)s {}'.format(dfx.__version__))
    parser.add_argument("videoPath", help="Path of video file to process")
    parser.add_argument("facePath", help="Path of face tracking data")
    parser.add_argument("studyPath", help="Path of study file")
    parser.add_argument("-o",
                        "--output",
                        help="Folder to save chunks",
                        default=None)
    args = parser.parse_args()

    doExtraction(args.videoPath, args.facePath, args.studyPath, args.output)
