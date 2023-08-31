import os
import csv
import glob
from tqdm import tqdm

def sort_tsv_files_by_onset_time(channel_folder, replace_existing=False):
    # Path to the 'Combined' input folder inside the selected channel folder
    input_folder = os.path.join(channel_folder, "Combined")

    # Get a list of all TXT files in the specified input folder
    tsv_files = glob.glob(os.path.join(input_folder, "*.txt"))

    # Path to the 'Combined_Sorted' output folder inside the selected channel folder
    output_folder = os.path.join(channel_folder, "Combined_Sorted")

    # If output folder exists and user wants to replace it, delete existing folder
    if os.path.exists(output_folder) and replace_existing:
        for existing_file in glob.glob(os.path.join(output_folder, "*")):
            os.remove(existing_file)
    elif os.path.exists(output_folder) and not replace_existing:
        print("Output folder already exists. Exiting without sorting.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each TSV file
    for tsv_file in tqdm(tsv_files, desc="Sorting files"):
        # Read the TSV data from the file
        with open(tsv_file, 'r') as file:
            tsv_data = list(csv.reader(file, delimiter='\t'))

            # Sort the TSV data by the onset time (assuming onset_time is in the second column)
            sorted_data = sorted(tsv_data, key=lambda row: float(row[1]))

        # Get the output file path within the 'Combined_Sorted' folder
        output_file = os.path.join(output_folder, os.path.basename(tsv_file))

        # Write the sorted data to the output TSV file
        with open(output_file, 'w', newline='') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            tsv_writer.writerows(sorted_data)

if __name__ == "__main__":
    channel = input("Enter the channel folder path: ")

    # Check if the output folder (channel/Combined_Sorted) already exists
    output_folder = os.path.join(channel, "Combined_Sorted")
    if os.path.exists(output_folder):
        replace_existing = input("The output folder already exists. Do you want to delete and replace it? (yes/no): ")
        if replace_existing.lower() == "yes":
            sort_tsv_files_by_onset_time(channel, replace_existing=True)
            print("TSV files sorted and saved to the output folder.")
        else:
            print("Sorting operation cancelled.")
    else:
        sort_tsv_files_by_onset_time(channel)
        print("TSV files sorted and saved to the output folder.")
