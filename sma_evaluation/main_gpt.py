import pandas as pd
import logging
import json
import os
import numpy as np
import time
import random
from openai import OpenAI
import re
from utils import read_yaml, read_json


config = read_json('gpt_config.json')

API_KEY = config["api_key"]
MODEL_NAME = config["model_name"]

client = OpenAI(api_key=API_KEY)

# ************************************************************

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH = 'dataset/'
transcript_dir = f'{PATH}/AMI_MS_Cleaned/'
related_docs_dir = f'{PATH}/truncated_single_input_agenda/'

# ************************************************************

# Load evaluation criteria from YAML
eval_criteria = read_yaml("config.yaml")["metrics"]["shared_docs"]

def save_df_to_csv(df, file_name):
    """
    Save a DataFrame to a CSV file.
    """
    df.to_csv(file_name, index=False)

def call_gpt(message, max_tokens):
    """
    Call the GPT-4 API to generate completions for the given message.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.01,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content.strip()  # type: ignore
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}") from e

def secure_model_call(prompt, base_delay=3.0, max_attempts=6, max_tokens=10000):
    attempt = 0
    ranking = "0"
    print("Calling model...", flush=True)
    while attempt < max_attempts:
        try:
            ranking = call_gpt(prompt, max_tokens)
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
    Parse the ranking response from the GPT-4 API.
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
    Build the prompt for the GPT-4 API based on the input data, supporting either transcript or related documents.
    
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
        task = """Your task is to rank the agenda based on the criteria provided.
                The agenda you evaluate should align with the given transcript and the target persona.
                Carefully assess whether the agenda meets the given criteria and fits the intended purpose.
                Remember to consider the quality of the agenda and how well it outlines the key discussion points of the meeting as reflected in the transcript.
                First, provide an argumentation for your ranking. Use chain-of-thought reasoning and evaluate step by step.
                Return a JSON object with the ranking for the evaluation criteria.
                The output should be in the following format:
                <explanation, step-by-step> \n\n ! \n\n <json object>
                The JSON object should follow the structure json \n {<evaluation criteria> : <Likert Score>} \n
                The JSON object should only contain the single Likert score for the currently assessed criteria.
                """
    else:  # source_type == "documents"
        task = """Your task is to evaluate the agenda based solely on the provided related documents and criteria.
                The agenda must align precisely with the content, objectives, and intent of the related documents, as interpreted for the target persona.
                For each step in the criteria:
                1. Explicitly compare every agenda item to the related documents.
                2. Identify matches (topics present in both), omissions (document topics missing from the agenda), and unsupported content (agenda topics not in the documents).
                3. Assess the severity of any misalignment using the provided scoring guidance (1 for severe, 2 for major, 3 for moderate, 4 for minor, 5 for perfect).
                Use chain-of-thought reasoning to justify your score step-by-step, addressing each criterion point explicitly.
                Assign a single Likert score from 1 (worst) to 5 (best) based on the cumulative alignment with the related documents.
                Return a JSON object with the ranking for the evaluation criteria.
                The output should be in the following format:
                <explanation, step-by-step> \n\n ! \n\n <json object>
                The JSON object should follow the structure json \n {<evaluation criteria> : <Likert Score>} \n
                The JSON object should only contain the single Likert score for the currently assessed criteria.
                """

    prompt = [
        {"role": "system", "content": f"{role}"},
        {"role": "user", "content": f"{material}\n\n{task}"},
    ]

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
            prompt = build_evaluation_prompt(transcript, agenda, description, persona, source_type="transcript")
        
        score = secure_model_call(prompt)
        logging.info("Score for criteria %s: %s", criteria, score)
        extracted_score = parse_ranking(score, criteria)
        
        for key, value in extracted_score.items():
            one_row_scores[key] = value

    return one_row_scores

def process_evaluation(args):
    """
    Process the evaluation with given arguments for source_dir and output_path.
    
    Args:
        args: Object containing source_dir and output_path attributes
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
    
    parser = argparse.ArgumentParser(description="Evaluate agendas based on transcripts and documents")
    parser.add_argument('--source-dir', type=str, default=f'{PATH}/truncated_single_input_agenda/', 
                       help='Directory containing source agenda files')
    parser.add_argument('--output-path', type=str, default='eval_output/', 
                       help='Path to save evaluation output')
    parser.add_argument('--output-csv', type=str, default='single_output.csv', 
                       help='Name of the output CSV file')
    parser.add_argument('--doctype', type=str, choices=['transcript', 'shared_docs'], default='transcript',
                        help='Document type to evaluate with agendas (must be "transcript" or "shared_docs")')
    
    args = parser.parse_args()
    process_evaluation(args)