import json
import pandas as pd
from pandas import json_normalize
import sys

def flatten_json(json_file, output_csv):
    """
    Function to flatten JSON structure and save it as a CSV file.
    
    Parameters:
    json_file (str): Path to the input JSON file.
    output_csv (str): Path to the output CSV file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Normalize the JSON data
    df = json_normalize(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Flattened data saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python flatten_json.py <input_json_file> <output_csv_file>")
    else:
        json_file = sys.argv[1]
        output_csv = sys.argv[2]
        flatten_json(json_file, output_csv)
