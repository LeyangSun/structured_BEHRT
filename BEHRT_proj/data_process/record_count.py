import os
import pandas as pd

# Define the path to the folder containing NCT CSV files
nct_folder = "/data/datasets/lou/Aokun_clinical_trial/"

# Define the path to the DIAGNOSES.csv file
diagnoses_file = "/data/datasets/leyang.sun/merged_age_diagnosis.csv"

# # Initialize a dictionary to store the counts for each NCT file
# nct_counts = {}

# # Loop through the NCT files in the specified folder
# for root, dirs, files in os.walk(nct_folder):
#     for file in files:
#         if file.startswith("NCT") and file.endswith(".csv"):
#             file_path = os.path.join(root, file)
            
#             # Read the CSV file and extract deid_pat_ID values
#             df = pd.read_csv(file_path)
#             deid_pat_ids = set(df["deid_pat_ID"])
            
#             # Read the DIAGNOSES.csv file
#             diagnoses_df = pd.read_csv(diagnoses_file)
            
#             # Count the number of deid_pat_ID values in DIAGNOSES.csv that match the current NCT file
#             count = len(set(diagnoses_df["deid_pat_ID"]).intersection(deid_pat_ids))
            
#             # Store the count in the dictionary
#             nct_counts[file] = count

# # Print the counts for each NCT file
# for file, count in nct_counts.items():
#     print(f"Number of deid_pat_ID in {file} that match DIAGNOSES.csv: {count}")





nct_folder = "/data/datasets/lou/Aokun_clinical_trial/"
selected_files = [
    'NCT00478205_26_ex_1.csv', 'NCT00478205_29_ex_5.csv', 'NCT00478205_29_ex_7.csv',
    'NCT00478205_24_ex_4.csv', 'NCT00478205_4_in_1.csv', 'NCT00478205_12_in_1.csv'
]
diagnoses_file = "/data/datasets/leyang.sun/merged_age_diagnosis.csv"
output_folder = "/data/datasets/leyang.sun/BEHRT_validation/"

for selected_file in selected_files:
    input_file = os.path.join(nct_folder, selected_file)
    output_file = os.path.join(output_folder, selected_file)

    # Read the selected file
    selected_data = pd.read_csv(input_file)

    # Read the diagnoses file
    diagnoses_data = pd.read_csv(diagnoses_file)

    # Extract records that match the deid_pat_ID in the selected file
    result = selected_data.merge(diagnoses_data, on='deid_pat_ID', how='inner')

    # Save the result to the output folder
    result.to_csv(output_file, index=False)