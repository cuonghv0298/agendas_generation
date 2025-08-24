import pandas as pd
import logging
import json
import os
import numpy as np
import time
import random
from google import genai  # Assuming this is the module you're using
import re
from utils import read_yaml

# ************************************************************
#                    SUPPORT FUNCTIONS                       #
# ************************************************************

def read_json(path):
    with open(path, encoding="utf8") as f:
        jsondict = json.load(f)
    return jsondict    

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH = 'dataset/'
transcript_dir = f'{PATH}/AMI_MS_Cleaned/'
related_docs_dir = f'{PATH}/truncated_single_input_agenda/'

# ************************************************************


def save_df_to_csv(df, file_name):
    """
    Save a DataFrame to a CSV file.
    """
    df.to_csv(file_name, index=False)

def call_gemini(prompt, max_tokens):
    """
    Call the Gemini API to generate completions for the given prompt.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,  # Using 'contents' as per your example
            # Assuming max_tokens or similar parameter exists; adjust if needed
        )
        return response.text.strip()
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}") from e

def secure_model_call(prompt, base_delay=3.0, max_attempts=6, max_tokens=1000000):
    attempt = 0
    ranking = "0"
    print("Calling model...", flush=True)
    while attempt < max_attempts:
        try:
            ranking = call_gemini(prompt, max_tokens)
            break
        except Exception as e:
            if "429" in str(e):  # Check if the exception is due to rate limiting
                sleep_time = (2 ** (attempt + 1)) + (random.randint(0, 1000) / 1000)
                logging.warning("Rate limit hit, backing off for %s seconds.", sleep_time)
                time.sleep(sleep_time)
                attempt += 1
            else:
                logging.error("Error encountered: %s", str(e))
                break
        finally:
            time.sleep(base_delay)
    return ranking

def parse_ranking(rankings, criteria="OOO"):
    """
    Parse the ranking response from the Gemini API.
    """
    match = re.search(r'```json\n({.*?})\n```', rankings, re.DOTALL)
    if match:
        json_string = match.group(1)
        json_object = json.loads(json_string)
        key, value = next(iter(json_object.items()))
        new_dict = {criteria: value}
        print(new_dict)
        return new_dict
    else:
        print("No JSON object found")
        json_object = {criteria: 0}
        print(json_object)
        return json_object

def build_evaluation_prompt(source, agenda, criteria, persona, source_type="transcript"):
    """
    Build the prompt for the Gemini API based on the input data, supporting either transcript or related documents.
    
    Args:
        source (str): The transcript or related documents content.
        agenda (str): The agenda to evaluate.
        criteria (str): The evaluation criteria (e.g., FAC, INF_DOC, etc.).
        persona (str): The target persona for the agenda.
        source_type (str): 'transcript' for transcript-based criteria, 'documents' for document-based criteria.
                          Defaults to 'transcript'.
    """
    role = """You are an expert in the field of meeting agendas and are tasked with evaluating the quality of the following agenda.
            Score the agenda according to the scoring criteria with a Likert score between 1 (worst) to 5 (best).
            Your evaluation must be based strictly on the provided materials and criteria, with no assumptions beyond the given content.
            """

    source_label = "Transcript" if source_type == "transcript" else "Related Documents"
    material = (
        f"{source_label}: <{source}>\n"
        f"Agenda: <{agenda}>\n"
        f"Criteria: <{criteria}>\n"
        f"Target Persona: <{persona}>\n"
    )

    if source_type == "transcript":
        task = """Your task is to evaluate the agenda based solely on the provided transcript and criteria.
                The agenda must align with the transcript’s content and the target persona’s perspective.
                Follow these steps:
                1. Provide a step-by-step explanation using chain-of-thought reasoning to justify your score.
                2. Assign a single Likert score from 1 (worst) to 5 (best) based on the criteria.
                3. Return your response in this exact format:
                   <Step-by-step explanation> \n\n ! \n\n ```json\n{"<criteria>": <score>}\n```
                Example output:
                   Step 1: <reasoning> \n Step 2: <reasoning> \n Score: 4 \n\n ! \n\n ```json\n{"FAC": 4}\n```
                Do not deviate from this format. Ensure the JSON object contains only the specified criteria and score.
                """
    else:  # source_type == "documents"
        task = """Your task is to evaluate the agenda based solely on the provided related documents and criteria.
                The agenda must align precisely with the content, objectives, and intent of the related documents.
                Follow these steps:
                1. Explicitly compare every agenda item to the related documents.
                2. Identify matches, omissions, and unsupported content.
                3. Assess misalignment severity (1=severe, 2=major, 3=moderate, 4=minor, 5=perfect).
                4. Provide a step-by-step explanation using chain-of-thought reasoning.
                5. Assign a single Likert score from 1 (worst) to 5 (best).
                6. Return your response in this exact format:
                   <Step-by-step explanation> \n\n ! \n\n ```json\n{"<criteria>": <score>}\n```
                Example output:
                   Step 1: <reasoning> \n Step 2: <reasoning> \n Score: 3 \n\n ! \n\n ```json\n{"FAC_DOC": 3}\n```
                Do not deviate from this format. Ensure the JSON object contains only the specified criteria and score.
                """

    # Combine into a single string for Gemini API
    prompt = f"{role}\n\n{material}\n\n{task}"
    return prompt

def compute_scores(transcript, related_docs, agenda, persona):
    """
    Compute the scores for the given transcript, related documents, and agenda.
    """
    one_row_scores = {}
    
    for criteria, description in eval_criteria.items():
        if criteria.endswith("_DOC"):
            # Use related documents for document-based criteria
            prompt = build_evaluation_prompt(related_docs, agenda, description, persona, source_type="documents")
        else:
            # Use transcript for transcript-based criteria
            print("Transcript: ", transcript)
            print("Agenda: ", agenda)
            # print("Agenda: ", agenda)
            prompt = build_evaluation_prompt(transcript, agenda, description, persona, source_type="transcript")
        
        score = secure_model_call(prompt)
        logging.info("Score for criteria %s: %s", criteria, score)
        extracted_score = parse_ranking(score, criteria)
        
        for key, value in extracted_score.items():
            one_row_scores[key] = value

    return one_row_scores

def process_evaluation(args):
    """
    Process the evaluation with given arguments for source_dir, output_path, and output_csv.
    
    Args:
        args: Object containing source_dir, output_path, and output_csv attributes
    """
    source_dir = args.source_dir
    out_path = args.output_path
    output_csv = args.output_csv
    
    if not os.path.exists(out_path): 
        os.makedirs(out_path)

    logger.info("Reading dataset...")

    all_dirs = os.listdir(source_dir)
    complete_dirs = []

    # dirs is all dirs without the complete dirs
    dirs = [i for i in all_dirs if i not in complete_dirs]
    print(len(dirs))

    output_csv_path = os.path.join(out_path, output_csv)
    header_written = False
    all_roles = set()

    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w') as f:
            pass

    # First pass to determine all unique roles
    for item in dirs:
        try:
            path = os.path.join(os.getcwd(), source_dir, item)
            with open(path, encoding="utf8") as f:
                jsondict = json.load(f)

            for participant in jsondict.get('Meeting Participants', []):
                role = participant.get('role', 'Unknown Role')
                all_roles.add(role)
        except Exception as e:
            logger.exception("Error occurred with file: %s", item)

    # Process each item in dirs
    for id, item in enumerate(dirs):
        try:
            path = os.path.join(os.getcwd(), source_dir, item)
            jsondict = read_json(path)
            transcript_path = os.path.join(os.getcwd(), transcript_dir, item)
            transcript_dict = read_json(transcript_path)
            related_docs_path = os.path.join(os.getcwd(), related_docs_dir, item)
            related_docs_dict = read_json(related_docs_path)
            
            agendas_list = [{'agenda': jsondict["agenda"]}] 
            transcript = transcript_dict["transcript"]
            related_docs = related_docs_dict["truncate_shared_docs"]
            role = 'Unknown Role'
            print(f"Processing {role}...")
            print(f"Transcript: {transcript[:100]}")
            persona = role
            for agenda_dict in agendas_list:
                agenda = agenda_dict.get('agenda', '')
                logger.info("***** Computing scores for %s for %s *****", role, item)
                score_pre = compute_scores(transcript, related_docs, agenda, persona)

                row = {
                    'Item': item,
                }
                
                # Initialize all role columns with empty strings or default values
                for r in all_roles:
                    row[f'agenda_{r}'] = ''
                    row[f'Score_{r}'] = ''

                # Assign the current role's agenda and score
                row[f'agenda_{role}'] = agenda
                row[f'Score_{role}'] = score_pre

                # Convert the row dictionary to a DataFrame
                row_df = pd.DataFrame([row])

                # Save to CSV, appending and avoiding header if already written
                with open(output_csv_path, 'a', encoding='utf8', newline='') as f:
                    row_df.to_csv(f, header=not header_written, index=False)
                    header_written = True

        except Exception as e:
            logger.exception("Error occurred with file: %s", item)

        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate agendas based on transcripts and documents using Gemini")
    parser.add_argument('--source-dir', type=str, default=f'{PATH}/truncated_single_input_agenda/', 
                       help='Directory containing source agenda files')
    parser.add_argument('--output-path', type=str, default='eval_output/', 
                       help='Path to save evaluation output')
    parser.add_argument('--output-csv', type=str, default='single_output.csv', 
                       help='Name of the output CSV file')
    parser.add_argument('--gemini-config', type=str, default='gemini_config.json', 
                       help='Name of the gemini configuration file')
    parser.add_argument('--doctype', type=str, choices=['transcript', 'shared_docs'], default='transcript',
                        help='Document type to evaluate with agendas (must be "transcript" or "shared_docs")')
    
    # Parse arguments
    args = parser.parse_args()

    # Extract Gemini configuration and document type
    gemini_config = args.gemini_config
    doctype = args.doctype
    
    # Load Gemini configuration
    config = read_json(gemini_config)
    # Load evaluation criteria from YAML
    eval_criteria = read_yaml("config.yaml")["metrics"][doctype]
    
    # Initialize Gemini client
    API_KEY = config["api_key"]
    MODEL_NAME = config["model_name"]
    client = genai.Client(api_key=API_KEY)
    
    # Run the evaluation process
    process_evaluation(args)