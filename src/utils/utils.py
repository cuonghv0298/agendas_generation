import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
import requests
import yaml
import re 
import tiktoken
import pandas as pd
import json

def chunking(method: str):
    """return TextSplitter with one chunking method"""
    if method == 'RecursiveCharacterTextSplitter':
        return RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    elif method == 'CharacterTextSplitter':
        return CharacterTextSplitter(
            separator=" ",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

def reformat_text(text):
    """Drop multiple spaces, tabs, endlines."""
    return " ".join(text.split())

def check_link_type(url):
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '').lower()

        if 'text/html' in content_type:
            return True #"This link is a webpage."
        else:
            return False #f"This link is not a webpage. Content-Type: {content_type}"

    except requests.RequestException as e:
        print(f'This link: {url} lead to result: {e}')
        return False
    
def find_links(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+')
    # Find all matches in the text
    links = re.findall(url_pattern, text)
    set_links = list(set(links))
    verified_links = [link for link in set_links if check_link_type(link)]
    return verified_links

def count_words(content):
    # Split the content by whitespace to count words
    words = content.split()
    # Return the length of the list of words
    return len(words)

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
def check_is_none(text: str) -> bool:
    return "none" in text.lower()

def preprocess_text_for_markdown(raw_text):
    # Replace newline characters with spaces
    cleaned_text = raw_text.replace("\n", " ")

    # Remove null character markers (\x00)
    cleaned_text = cleaned_text.replace("\x00", "")

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Use regex to identify bullet points (●) and ensure proper formatting
    cleaned_text = re.sub(r"●", "\n- ", cleaned_text)
    
    return cleaned_text

def clean_text(raw_text):
     # Replace multiple consecutive newlines (\n\n, \n\n\n, etc.) with a single \n
    cleaned_text = re.sub(r'\n{2,}', '\n', raw_text)
    # Remove extra spaces at the beginning of each line
    cleaned_text = re.sub(r'^ +', '', cleaned_text, flags=re.MULTILINE)
    return cleaned_text.strip()  # Remove leading/trailing spaces

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_text_from_txt(jsondict):
    outputs = []
    for x in jsondict['shared-doc']['txt']:
        filename = x['filename']
        content = x['content']
        o = f'{filename} \n ------------ \n {content}'
        outputs.append(o)
    outputs = '\n'.join(outputs)
    return outputs

def get_text_from_doc(jsondict):
    outputs = []
    for x in jsondict['shared-doc']['doc']:
        filename = x['filename']
        content = x['content']
        o = f'{filename} \n ------------ \n {content}'
        outputs.append(o)
    outputs = '\n'.join(outputs)
    return outputs

def get_text_from_ppt(jsondict):
    outputs = []
    for x in jsondict['shared-doc']['ppt']:
        filename = x['filename']
        string = ['\n'.join(v) for k,v in x['content'].items()]
        content = '\n'.join(string)
        o = f'{filename} \n ------------ \n {content}'
        outputs.append(o)
    outputs = '\n'.join(outputs)
    return outputs 

def truncate_shared_docs(shared_docs: str, max_tokens: int = 90000) -> str:
    """
    Truncate the shared documents to ensure the number of tokens is less than max_tokens.
    Args:
        shared_docs (str): The shared documents as a string.
        max_tokens (int): The maximum number of tokens allowed.
    Returns:
        str: The truncated shared documents.
    """
    # Ensure the shared documents are within the token limit
    while num_tokens_from_string(shared_docs) > max_tokens:
        shared_docs = shared_docs[:-1000]  # Truncate the last 1000 characters iteratively
    return shared_docs

def load_data_with_shared_doc_path(path: str = "EDA/token_data.csv") -> pd.DataFrame:
    """
    Load the data from the path
    """
    df = pd.read_csv(path)
    df_shared_docs = df[df["num_tokens_shared_doc"] > 0]
    return df_shared_docs

def extract_data_from_file(file: str, root: str = '/datadrive/CuongHV/project/DATA/AMI_MS_Cleaned') -> pd.DataFrame:
    """
    Extract the data from the file
    """
    path = f'{root}/{file}'
    with open(path, encoding='utf-8') as f:
        jsondict = json.load(f)
    return jsondict
def extract_category(filename: str):
    key = filename[-6]
    if key == 'a':
        category = 'Project Kick-off'
        description = "Consisting of building a project team and getting acquainted with each other and the task."
    elif key == 'b':
        category = 'Functional design'
        description = "In which the team sets the user requirements, the technical functionality, and the working design."
    elif key == 'c':
        category = 'Conceptual design'
        description = "Which finalizes the look-and-feel and user interface, and in which the result is evaluated."
    elif key == 'd':
        category = 'Detailed design'
        description = "Which finalizes the look-and-feel and user interface, and in which the result is evaluated."
    else:
        raise "Another key in Filename, plaease check it"
    return category, description