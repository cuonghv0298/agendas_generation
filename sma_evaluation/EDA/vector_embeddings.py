from typing import Dict, Any
import json
import time
import random
import argparse
import os
import pandas as pd
from google import genai

def read_json(file_path: str) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    with open(file_path, encoding="utf8") as f:
        return json.load(f)

def sleep_random(min_sleep: float, range_sleep: float) -> None:
    """Sleep for a random duration between min_sleep and min_sleep + range_sleep."""
    time.sleep(min_sleep + random.random() * range_sleep)

def process_files(gemini_config: str, source_dir: str, output_csv: str) -> None:
    """Process files and generate embeddings using Gemini API."""
    # Load configuration
    config = read_json(gemini_config)
    API_KEY = config["api_key"]
    client = genai.Client(api_key=API_KEY)

    # Get list of files
    dirs = os.listdir(source_dir)

    # Prepare data storage
    data = {
        "Item": [],
        "Embedding_Vector": []
    }

    for id, item in enumerate(dirs):
        try:
            path = os.path.join(os.getcwd(), source_dir, item)
            jsondict = read_json(path)
            agenda = jsondict["agenda"]

            # Generate embeddings
            result = client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=agenda)
            # print(result.embeddings[0].values)

            # Store results
            data["Item"].append(item)
            data["Embedding_Vector"].append(result.embeddings[0].values)
            sleep_random(15, 20)

            # Create DataFrame
            df = pd.DataFrame(data)

            # Define file paths
            output_dir = os.path.dirname(output_csv)
            filename = os.path.basename(output_csv)
            output_base, output_ext = os.path.splitext(filename)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            new_output_path = output_csv if id == len(dirs) - 1 else os.path.join(output_dir, f"{output_base}_{id}{output_ext}")
            old_output_path = os.path.join(output_dir, f"{output_base}_{id-1}{output_ext}") if id > 0 else None

            # Save new CSV
            df.to_csv(new_output_path, index=False)
            print(f"Embeddings saved to {new_output_path}")

            # Remove old temporary file if it exists and new file was created successfully
            if old_output_path and os.path.exists(old_output_path) and os.path.exists(new_output_path):
                os.remove(old_output_path)
                print(f"Removed old file: {old_output_path}")

        except Exception as e:
            print(f"Error occurred with file {item}: {str(e)}")

    # Display first few rows
    print("\nFirst few rows of the saved data:")
    print(df.head())

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Process files and generate embeddings using Gemini API")
    parser.add_argument(
        "--gemini-config",
        type=str,
        required=True,
        help="Path to the Gemini configuration JSON file"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing source JSON files"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to the output CSV file for final embeddings"
    )

    args = parser.parse_args()
    process_files(args.gemini_config, args.source_dir, args.output_csv)

if __name__ == "__main__":
    main()