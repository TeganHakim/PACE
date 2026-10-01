"""
Find and remove corrupt samples in VisDrone-DroneVehicle.

    python dataset_tools.py check  [root]           # scan images, write bad_images.txt
    python dataset_tools.py delete [root] [--yes]   # delete every sample listed in it

A sample is the RGB image, IR image, RGB label and IR label that share a
filename stem. If any image is corrupt, all four are deleted so no
incomplete sample is left behind.
"""

import argparse
import os
from os.path import isdir, join, splitext

import cv2

DEFAULT_ROOT = "../VisDrone-DroneVehicle"
SPLITS = ("train", "val", "test")
IMG_EXTS = ".jpg"


def bad_file_path(root):
    return join(root, "bad_images.txt")


# ---------------------------------------------------------------------------
# 1. check
# ---------------------------------------------------------------------------
def check(root):
    bad = set()

    for split in SPLITS:
        for suffix in ("", "r"):  # "" = rgb, "r" = infrared
            folder = join(root, split, split + "img" + suffix)
            if not isdir(folder):
                print("Missing folder:", folder)
                continue

            files = sorted(
                f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)
            )
            print("Checking {} ({} files)".format(folder, len(files)))

            for f in files:
                path = join(folder, f)
                if (
                    os.path.getsize(path) == 0
                    or cv2.imread(path, cv2.IMREAD_GRAYSCALE) is None
                ):
                    print("  BAD:", path)
                    bad.add("{}/{}".format(split, f))

    with open(bad_file_path(root), "w") as fh:
        fh.write("\n".join(sorted(bad)) + ("\n" if bad else ""))

    print(
        "Done. {} bad image name(s) written to {}".format(len(bad), bad_file_path(root))
    )


# ---------------------------------------------------------------------------
# 2. delete
# ---------------------------------------------------------------------------
def build_index(folder):
    """Map filename stem -> list of full paths in that folder."""
    index = {}
    if isdir(folder):
        for f in os.listdir(folder):
            index.setdefault(splitext(f)[0], []).append(join(folder, f))
    return index


def delete(root, yes=False):
    bad_file = bad_file_path(root)
    if not os.path.exists(bad_file):
        raise FileNotFoundError("{} not found. Run 'check' first.".format(bad_file))

    # lines look like "train/12901.jpg"
    bad = {}
    with open(bad_file) as fh:
        for line in fh.read().split():
            split, name = line.split("/", 1)
            bad.setdefault(split, set()).add(splitext(name)[0])

    # collect everything to delete
    to_delete = []
    samples = 0

    for split in SPLITS:
        if split not in bad:
            continue

        folders = {
            "RGB image": join(root, split, split + "img"),
            "IR image": join(root, split, split + "imgr"),
            "RGB label": join(root, split, split + "label"),
            "IR label": join(root, split, split + "labelr"),
        }
        indexes = {part: build_index(folder) for part, folder in folders.items()}

        for stem in sorted(bad[split]):
            samples += 1
            print("{}/{}".format(split, stem))

            for part, index in indexes.items():
                paths = index.get(stem, [])
                if not paths:
                    print("    {:<10} not found (already gone)".format(part))
                for path in paths:
                    print("    {:<10} {}".format(part, path))
                    to_delete.append(path)

    if not to_delete:
        print("\nNothing to delete.")
        return

    print(
        "\n{} files across {} samples will be deleted.".format(len(to_delete), samples)
    )

    if not yes and input("Type 'yes' to continue: ").strip().lower() != "yes":
        print("Cancelled, nothing deleted.")
        return

    for path in to_delete:
        os.remove(path)

    os.rename(bad_file, bad_file + ".done")
    print(
        "Deleted {} files. Renamed bad_images.txt to bad_images.txt.done".format(
            len(to_delete)
        )
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and remove corrupt DroneVehicle samples."
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    p_check = sub.add_parser("check", help="scan images and write bad_images.txt")
    p_check.add_argument("root", nargs="?", default=DEFAULT_ROOT)

    p_delete = sub.add_parser(
        "delete", help="delete all four parts of every sample in bad_images.txt"
    )
    p_delete.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    p_delete.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )

    args = parser.parse_args()

    if args.command == "check":
        check(args.root)
    else:
        delete(args.root, yes=args.yes)
