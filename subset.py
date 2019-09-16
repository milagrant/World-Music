import os
import json
import shutil


def copy_rename(name, src_dir):
    """
    Copy audio file to audio folder and rename it in accordance with audiolist.txt
    """
    old_name = "SampleAudio.mp3"
    dst_dir = "data/audio"
    src_file = os.path.join(src_dir, old_name)
    shutil.copy(src_file, dst_dir)

    dst_file = os.path.join(dst_dir, old_name)
    new_dst_name = os.path.join(dst_dir, name)
    os.rename(dst_file, new_dst_name)


def main():
    """
    Find new names of audio files based on catalog number in metadata.
    """
    for folder in os.listdir("smithsonian"):
        if os.path.isdir("smithsonian/" + folder):
            for subfolder in os.listdir("smithsonian/" + folder):
                if os.path.isdir("smithsonian/" + folder + "/" + subfolder):
                    for filename in os.listdir("smithsonian/" + folder + "/" + subfolder):
                        with open("smithsonian/" + folder + "/" + subfolder + "/" + "metadata.json") as json_file:
                            data = json.load(json_file)
                            new_name = data["CatalogNumber"] + ".mp3"
                            copy_rename(new_name, "smithsonian/" + folder + "/" + subfolder)


if __name__ == "__main__":
    main()