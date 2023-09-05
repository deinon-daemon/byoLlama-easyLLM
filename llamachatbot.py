import os
from easyllm.clients import huggingface
import re

class LlamaChatBot:

    def __init__(self):
        self.prompt_builder = "llama2"
        self.api_key = os.environ.get("HF_TOKEN")
        huggingface.api_key = self.api_key

    def get_last_sentence(self, text):
      if text.endswith('.'):
        return text
      
      # Find last sentence
      sentences = re.split(r'[.!?]\s*', text)
      last_sentence = sentences[-1]
      
      # Get text up until last sentence
      text = text[:text.rindex(last_sentence)]
      
      return text    

    def clean_text(self, text):
      # Replace newlines, tabs, and multiple spaces with single space
      cleaned = re.sub(r'[\n\t]+', ' ', text)
      cleaned = re.sub(r' +', ' ', cleaned)
      
      # Condense whitespace down to just two spaces
      cleaned = re.sub(r' {2,}', '  ', cleaned)

      return cleaned

    def get_chat_response(self, messages, temperature=0.9, top_p=0.6, max_tokens=256):
        response = huggingface.ChatCompletion.create(
            model="meta-llama/Llama-2-70b-chat-hf", ##"meta-llama/Llama-2-70b-chat-hf", togethercomputer/LLaMA-2-7B-32K, amazon/FalconLite, stabilityai/StableBeluga2, https://huggingface.co/upstage/Llama-2-70b-instruct-v2
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

    def get_sum_response(self, name, context, temperature=0.5, top_p=0.6, max_tokens=256):
        summarizer_chat_prompt = f'''You are a helpful summarizer Llama-LLM. Your job is to write a detailed description / profile of {name} based on the text provided by the user.
        In your description, you should touch on any of the following properties of {name}: its purpose, mission, programs, offerings, resources, industries, audiences, etc.
        Be eloquent and fairly succinct. This should be like a business profile or a short, informative news blurb that summarizes the text content with an emphasis on the subject, {name}.'''
        summarizer_user_prompt = f'''Write a description for {name}. Here is the source text: \n\n {context}'''  
        messages = [{'role': 'system', 'content': summarizer_chat_prompt}, {'role': 'user', 'content': summarizer_user_prompt}]
        return self.get_chat_response(messages, temperature, top_p, max_tokens) 

    def condense_summary(self, name, text, temperature=0.5, top_p=0.6, max_tokens=150):
        summarizer_chat_prompt = f'''You are a helpful summarizer Llama-LLM. Your job is to condense / summarize whatever text is given to you. 
        Your response should be no longer than 2 sentences.'''

        user_prompt = f'''Summarize / condese the following description for {name}. Be succinct. Here is the description to shorten: {text}.'''
        messages = [{'role': 'system', 'content': summarizer_chat_prompt}, {'role': 'user', 'content': user_prompt}]
        return self.get_chat_response(messages, temperature, top_p, max_tokens)    


    def get_summaries(self, name, text, temperature=0.5, top_p=0.6, max_tokens=256):
        context = self.clean_text(text)
        if len(context) > 15000:
          context = context[:13999]
        long_des = self.get_sum_response(name, context)
        short_des = self.condense_summary(name, long_des)
        return {'long_description': self.get_last_sentence(long_des), 'short_description': short_des} 
 