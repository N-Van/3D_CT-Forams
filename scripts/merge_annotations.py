import click
import pandas as pd
import json
from pathlib import Path


def get_index(filename):
    return int(filename.split("_")[-1])


def s2i(value):
    if isinstance(value, str):
        value = value.replace(",", ".")
    return int(float(value))


def compute_distance(x1, y1, z1, x2, y2, z2):
    import math

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def duplicate_wrapper(duplicate, radius):
    def row_wrapper(row):
        x1, y1, z1 = duplicate
        x2, y2, z2 = s2i(row["Xcoords"]), s2i(row["Ycoords"]), s2i(row["Zcoords"])
        return compute_distance(x1, y1, z1, x2, y2, z2) > radius

    return row_wrapper


def remove_wrong_fn(csv_data, duplicates_to_remove, radius=8):
    for duplicate in duplicates_to_remove:
        f = duplicate_wrapper(duplicate, radius=radius)
        csv_data = csv_data[csv_data.apply(f, axis=1)]
    return csv_data


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

    slice_shape = 250
    # slice_shape = 475

    filtered_columns = ["Unnamed: 0.1", "Unnamed: 0"]

    duplicates_to_remove = []
    added_count = 0

    # Iterate through json data and add to csv data
    for data in error_json:
        frame_index = get_index(data["file"])
        x, y, z = data["x"], data["y"], data["z"] + (slice_shape * frame_index)
        if data["type"] == "FP" and data["corrected"] == 0:
            # x, y, z = data["x"], data["y"] + (slice_shape * frame_index), data["z"]
            new_data = pd.DataFrame({"Xcoords": [x], "Ycoords": [y], "Zcoords": [z]})
            csv_data = pd.concat([csv_data, new_data], ignore_index=True)
            added_count += 1
        if data["type"] == "FN" and data["corrected"] == 0:
            duplicates_to_remove.append((x, y, z))

    print(f"Added #{added_count} annotations")
    print(f"Len csv_data before cleansing = {len(csv_data)}")
    print(f"Should remove #{len(duplicates_to_remove)} rows")
    csv_data = remove_wrong_fn(csv_data, duplicates_to_remove)
    print(f"New csv_data length = {len(csv_data)}")

    for filter_col in filtered_columns:
        if filter_col in csv_data.columns:
            csv_data = csv_data.drop(columns=[filter_col])

    # Save csv
    csv_data.to_csv(f"{csv_output_path}", sep=";")


if __name__ == "__main__":
    main()
