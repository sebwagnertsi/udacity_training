#!/usr/bin/env python
import argparse
import logging
import pathlib
import wandb
import requests
import tempfile


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Derive the base name of the file from the URL
    basename = pathlib.Path(args.file_url).name.split("?")[0].split("#")[0]

    # Download file, streaming so we can download files larger than
    # the available memory. We use a named temporary file that gets
    # destroyed at the end of the context, so we don't leave anything
    # behind and the file gets removed even in case of errors
    logger.info(f"Downloading {args.file_url} ...")


    #### CAVEAT: Running the project on windows does not allow the file to be opened for writing and reading it at the same time.
    # So i do open my own temp file here and close it before doint the wandb upload stuff.
    windows_tmpfile = ""
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        windows_tmpfile = fp.name   # get a cool temp file name in a proper tmp folder
    
    with open(windows_tmpfile, 'wb+') as fp:
        # Download the file streaming and write to open temp file
        with requests.get(args.file_url, stream=True) as r:
            for chunk in r.iter_content(chunk_size=8192):
                fp.write(chunk)

            # Make sure the file has been written to disk before uploading
            # to W&B
        fp.flush()

    logger.info("Creating run exercise_2")
    with wandb.init(project="exercise_2", job_type="download_data") as run:
        logger.info("Creating artifact")
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description,
            metadata={'original_url': args.file_url}
        )
        artifact.add_file(windows_tmpfile, name=basename)

        logger.info("Logging artifact")
        run.log_artifact(artifact)

        # This makes sure that the artifact is uploaded before the
        # tempfile is destroyed
        artifact.wait()

    # Remove my own temp file:
    import os
    try:
        os.remove(windows_tmpfile)
    except OSError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
