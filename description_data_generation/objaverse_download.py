import random
import multiprocessing as mp
import json
import os
import objaverse
import pandas as pd
import objaverse.xl as oxl


def download_objaverse(xl = False):
    random.seed(32)

    if xl:
        annotations = oxl.get_annotations()

        #filter out the objects that are not .glb

        filtered = annotations[annotations['fileType'] == 'glb']


        oxl.download_objects(
            objects=filtered,
            download_dir="./objaverse_xl",
            processes=mp.cpu_count(),
            overwrite=False,
            skip_existing=True,
        )

        # save the metadata to a JSON file
        output_dir = "./"
        # export the filtered annotations to a CSV file
        output_file = os.path.join(output_dir, "objaverse_metadata.csv")
        filtered.to_csv(output_file, index=False)
        print(f"Saved metadata to {output_file}")

    else:

        uids = objaverse.load_uids()


        random_object_uids = random.sample(uids, 10000)


        # Load metadata for these objects
        annotations = objaverse.load_annotations(random_object_uids)

        objects = objaverse.load_objects(
            uids=random_object_uids,
            download_processes=mp.cpu_count())

        # save the uids and file paths to a JSON file
        output_dir = "./"
        output_file = os.path.join(output_dir, "objaverse_metadata.json")
        with open(output_file, 'w') as f:
            json.dump({
                "objects": objects
            }, f, indent=4)
        print(f"Saved metadata to {output_file}")


if __name__ == "__main__":
    # get the arguments from the command line
    import argparse

    parser = argparse.ArgumentParser(description="Download Objaverse dataset.")
    parser.add_argument(
        "--xl",
        action="store_true",
        help="Download the Objaverse XL dataset. Instead of the regular Objaverse dataset.",
    )
    args = parser.parse_args()
    download_objaverse(xl=args.xl)



