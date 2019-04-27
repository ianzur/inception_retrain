"""
Python script to aid in labeling frames of videos for SPECIFIC training with a pre-trained classifier

Randomly parses through percentage, "per", of videos in "video_directory"
You may add labels as you proceed (not efficient for >10 classes)

When an image appears press a key.
That key is bound to the class you define next. (Image Window title will remind you of current keys)
images are numbered ascending from 1.jgp
and placed in a folder "images/class"

Sometimes Image window does not pop to front

Dependants opencv (pip install python-opencv)

Ian Zurutuza
"""
import cv2
import os
import random

video_directory = ""
output_directory = "images"

per = 0.01                      # percentage of each video to label
key_dict = {}                   # used for controlling where images are saved
label_string = "My labels: "    # used to remind me the controls
total = 0                       # used to number images

def label(file: str, path: str):
    """
    :param file: specific video we are going to label
    :param path: output path
    :return: nada

    Open the video randomly parse through % of video while labeling images
    Saving each image to the folder specifying the label name.

    you may remove the random bit (just 2 lines at the beginning of the for loop)
    """
    global key_dict
    global label_string
    global total

    print("labeling: ", file)

    vid = cv2.VideoCapture(file)

    # TODO: add calculation to get average frame rate & trial length
    # print(vid.get(cv2.CAP_PROP_FPS))
    # print(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(int(per * vid.get(cv2.CAP_PROP_FRAME_COUNT))):

        # pick random frame
        random_frame = random.randint(0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
        vid.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        # get rid of these ^ two lines ^ if you don't want random

        ret, frame = vid.read()

        cv2.namedWindow(label_string)
        cv2.moveWindow(label_string, 40, 30)
        cv2.imshow(label_string, frame)

        key = cv2.waitKey(0)    # wait for any key press

        total += 1

        # save image to existing label
        if key in key_dict:
            p = os.path.join(path, key_dict[key], str(total) + ".jpg")

            if not cv2.imwrite(p, frame):
                raise IOError("failed to save {}".format(p))
            else:
                print("image saved to {} \n".format(p))

        else:   # create new label
            name = input("Enter Label Name: ")

            cv2.destroyWindow(label_string)
            key_dict[key] = name                # add new key to key_dict
            os.mkdir(os.path.join(path, name))
            label_string = label_string + chr(key) + "=" + name + "  "      # append label string for window title

            p = os.path.join(path, key_dict[key], str(total) + ".jpg")

            if not cv2.imwrite(p, frame):
                raise IOError("failed to save {}".format(p))
            else:
                print("image saved to {} \n".format(p))

    vid.release()
    cv2.destroyAllWindows()
    return


def main():
    global key_dict
    global label_string
    global total

    random.seed()

    out_path = os.path.join(os.getcwd(), output_directory)

    # check if labeled images exist and if so append
    if os.path.exists(out_path):
        print("\n{} exists\nTallying already labeled images".format(out_path))
        for root, dirs, files in os.walk(out_path):
            for d in dirs:
                k = input("\nEnter key to bind to class {}: ".format(d))
                print(k, ord(k))
                key_dict[ord(k)] = d
                label_string = label_string + k + "=" + d + "  "

            for file in files:
                if file.endswith(".jpg"):
                    total += 1
    else:   # make directory
        os.mkdir(out_path)

    # for each raw video
    for root, dirs, files in os.walk(video_directory):
        for file in files:

            # double check this is a video
            # TODO: add different types of videos
            if file.endswith(".mp4"):

                label(os.path.join(root, file), out_path)

    print("total images classified = {}".format(total))
    return


if __name__ == '__main__':
    main()
