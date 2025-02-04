import click
import pandas as pd
import json
from pathlib import Path


def get_index(filename):
    return int(filename.split("_")[-1])


@click.command()
@click.option(
    "--csv_path",
    required=True,
)
@click.option(
    "--error_json",
    required=True,
)
@click.option(
    "--csv_output_path",
    required=True,
)
def main(csv_path, error_json, csv_output_path):
    # Load csv
    csv_data = pd.read_csv(csv_path, sep=";")

    # Load json
    with open(error_json, "r") as fp:
        error_json = json.load(fp)

    # slice_shape = 250
    # slice_shape = 475
    slice_shape = 0

    filtered_columns = ["Unnamed: 0.1", "Unnamed: 0"]

    # Iterate through json data and add to csv data
    for data in error_json:
        if data["corrected"] == 0:
            frame_index = get_index(data["file"])
            # x, y, z = data["x"], data["y"], data["z"] + (slice_shape * frame_index)
            x, y, z = data["x"], data["y"] + (slice_shape * frame_index), data["z"]
            new_data = pd.DataFrame({"Xcoords": [x], "Ycoords": [y], "Zcoords": [z]})
            csv_data = pd.concat([csv_data, new_data], ignore_index=True)

    for filter_col in filtered_columns:
        if filter_col in csv_data.columns:
            csv_data = csv_data.drop(columns=[filter_col])

    # Save csv
    csv_data.to_csv(f"{csv_output_path}", sep=";")


if __name__ == "__main__":
    main()
