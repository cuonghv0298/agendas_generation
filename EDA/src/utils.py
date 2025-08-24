import re 
import tiktoken

def clean_text(raw_text):
     # Replace multiple consecutive newlines (\n\n, \n\n\n, etc.) with a single \n
    cleaned_text = re.sub(r'\n{2,}', '\n', raw_text)
    # Remove extra spaces at the beginning of each line
    cleaned_text = re.sub(r'^ +', '', cleaned_text, flags=re.MULTILINE)
    return cleaned_text.strip()  # Remove leading/trailing spaces

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = "cl100k_base"
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