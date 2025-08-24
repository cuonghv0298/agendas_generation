from langchain.prompts import PromptTemplate
from langchain import callbacks

class Simple_Assistance:
    """
    From the defined template and input content to generate text
    Arg:
    - template: prompt confige
    - content: text input
    """

    def __init__(
        self,
        template,
        content,
        llm,
    ):
        prompt = PromptTemplate.from_template(
            template
        )
        prompt.format(
            content=content,
        )
        self.llm_chain = prompt | llm
        self.content =  content
        
    def __call__(self):        
        with callbacks.collect_runs() as cb:
            result = self.llm_chain.invoke(
                {
                    "content": self.content,
                },
            )
            run_id= cb.traced_runs[0].id
        response = {
                "text": result.content,
                "__run":{
                    "run_id": run_id
                }
            }
        return response

class Two_Input_Assistance:
    """
    From the defined template and input content to generate text
    Arg:
    - template: prompt confige
    - input 1: text input
    - input 2: text input
    """

    def __init__(
        self,
        template,
        input1,
        input2,
        llm,
    ):
        prompt = PromptTemplate.from_template(
            template
        )
        prompt.format(
            input1=input1,
            input2=input1,
        )
        self.llm_chain = prompt | llm
        self.input1 =  input1
        self.input2 = input2
        
    def __call__(self):        
        with callbacks.collect_runs() as cb:
            result = self.llm_chain.invoke(
                {
                    "input1": self.input1,
                    "input2": self.input2,
                },
            )
            run_id= cb.traced_runs[0].id
        response = {
                "text": result.content,
                "__run":{
                    "run_id": run_id
                }
            }
        return response

class Triple_Input_Assistance:
    """
    From the defined template and input content to generate text
    Arg:
    - template: prompt confige
    - input 1: text input
    - input 2: text input
    - input 3: text input
    """

    def __init__(
        self,
        template,
        input1,
        input2,
        input3,
        llm,
    ):
        prompt = PromptTemplate.from_template(
            template
        )
        prompt.format(
            input1=input1,
            input2=input1,
            input3=input3,
        )
        self.llm_chain = prompt | llm
        self.input1 =  input1
        self.input2 = input2
        self.input3 = input3

    def __call__(self):        
        with callbacks.collect_runs() as cb:
            result = self.llm_chain.invoke(
                {
                    "input1": self.input1,
                    "input2": self.input2,
                    "input3": self.input3,
                },
            )
            run_id= cb.traced_runs[0].id
        response = {
                "text": result.content,
                "__run":{
                    "run_id": run_id
                }
            }
        return response
    