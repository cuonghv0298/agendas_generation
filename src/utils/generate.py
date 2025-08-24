from src.service.genbot import Two_Input_Assistance, Simple_Assistance, Triple_Input_Assistance

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

class Generation:
    def __init__(
        self,
        prompt_config
    ):
        self.prompt_config = prompt_config
        self.default_param = {
            "temperature":0.7,
            "top_p":0.8, 
            "max_retries":2,
        }
    def generate_recap_agenda(
        self,
        llm, 
        transcript,
        summary, 
        prompt="",
        params = {},        
    ):
        # Set model para
        if params == {}:
            params = self.default_param
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_recap_agenda_prompt']
        
        # Set llm para
        assistant_param = {
            "template": prompt,
            "input1": transcript,
            "input2": summary,
            "llm": llm, 
        }

        bot = Two_Input_Assistance(**assistant_param)
        response = bot()
        return response
    def generate_agenda_template(
            self,
            llm,
            agendas,
            prompt="",
            params = {},
    ):
        if params == {}:
            params = self.default_param
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_agenda_template_prompt']
        
        # Set llm para
        assistant_param = {
            "template": prompt,
            "content": agendas,
            "llm": llm, 
        }

        bot = Simple_Assistance(**assistant_param)
        response = bot()
        return response
    def generate_truncated_sigle_input_agenda(
            self,
            llm,
            shared_docs,
            prompt="",
            params = {},
    ):
        if params == {}:
            params = self.default_param
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_truncated_sigle_input_agenda_prompt']
        
        # Set llm para
        assistant_param = {
            "template": prompt,
            "content": shared_docs,
            "llm": llm, 
        }

        bot = Simple_Assistance(**assistant_param)
        response = bot()
        return response
    def generate_truncated_multi_input_agenda(
            self,
            llm,
            shared_docs,
            prompt="",
            params = {},
    ):
        if params == {}:
            params = self.default_param
        
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_truncated_multi_input_agenda_prompt']
        # Set llm para
        assistant_param = {
            "template": prompt,
            "content": shared_docs,
            "llm": llm, 
        }

        bot = Simple_Assistance(**assistant_param)
        response = bot()
        return response
    def generate_category_truncated_multi_input_agenda(
            self,
            llm,
            shared_docs,
            category,
            description,
            prompt="",
            params = {},
    ):
        if params == {}:
            params = self.default_param
        
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_category_truncated_multi_input_agenda_prompt']
        # Set llm para
        assistant_param = {            
            "template": prompt,
            "input1": category,
            "input2": description,
            "input3": shared_docs,
            "llm": llm, 
        }

        bot = Triple_Input_Assistance(**assistant_param)
        response = bot()
        return response
    
    def generate_rag_multi_input_agenda(
            self,
            llm,
            qa_text,
            prompt="",
            params = {},
    ):
        if params == {}:
            params = self.default_param
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_rag_multi_input_agenda_prompt']
        
        # Set llm para
        assistant_param = {
            "template": prompt,
            "content": qa_text,
            "llm": llm, 
        }

        bot = Simple_Assistance(**assistant_param)
        response = bot()
        return response
    
    def generate_category_rag_multi_input_agenda(
            self,
            llm,
            category,
            description,
            qa_text,
            prompt="",
            params = {},
    ):
        if params == {}:
            params = self.default_param
        
        # Set prompt
        if prompt == "":        
            prompt = self.prompt_config['prompt']['generate_category_rag_multi_input_agenda_prompt']
        
        # Set llm para
        assistant_param = {
            "template": prompt,
            "input1": category,
            "input2": description,
            "input3": qa_text,
            "llm": llm, 
        }

        bot = Triple_Input_Assistance(**assistant_param)
        response = bot()
        return response
    def summarized_by_stuff(
            self,
            llm,
            docs
        ):
        
        prompt_template = self.prompt_config['prompt']["summarized_by_stuff_prompt"]
        prompt = PromptTemplate.from_template(prompt_template)
        stuff_chain = create_stuff_documents_chain(
                llm=llm, 
                prompt = prompt,
                document_variable_name = "text",
                )
        return stuff_chain.invoke({"text": docs})


    