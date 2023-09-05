import os
import re
import json
import boto3
import torch
import logging
import sagemaker
import dirtyjson
from flask import jsonify
from huggingface_hub import login
from easyllm.clients import huggingface
from typing import List, Union, Dict, Any
from sagemaker.huggingface import HuggingFaceModel

session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_DEFAULT_REGION')
)
sess = sagemaker.Session(session)

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message.content}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt   


class LlamaJsonformer:

  def __init__(
        self,
        json_schema: Dict[str, Any],
        prompt: str = "Generate factual information as json based on the following schema and context:",
        context: str = 'Use your imagination to populate the json schema!',
        model: str = 'meta-llama/Llama-2-70b-chat-hf',
        sagemaker_model: str = 'huggingface-pytorch-tgi-inference-2023-08-15-20-03-54-414',
        max_number_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = .6,
        max_string_token_length: int = 256,
    ):

    self.text = ''
    self.model = model
    self.sagemaker_model = sagemaker_model
    self.llm = sagemaker.huggingface.model.HuggingFacePredictor(self.sagemaker_model, sess)
    self.json_schema = json_schema
    self.prompt = prompt
    self.context = context

    self.max_number_tokens = max_number_tokens
    self.temperature = temperature
    self.top_p = top_p

    self.max_string_token_length = max_string_token_length

    # helper to build llama2 prompt
    huggingface.prompt_builder = "llama2"
    login(os.environ.get('HF_TOKEN'))


  def get_json_start(self, text):
      start = text.find('{')
      return text[start:]

  def get_json_end(self, text):
      end = text.rfind('}')
      return text[:end+1]

  def clean_text(self, text):
    # Replace newlines, tabs, and multiple spaces with single space
    cleaned = re.sub(r'[\n\t]+', ' ', text)
    cleaned = re.sub(r' +', ' ', cleaned)
    
    # Condense whitespace down to just two spaces
    cleaned = re.sub(r' {2,}', '  ', cleaned)

    return cleaned

  def falcon(self):

    template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request in the format requested.

    ### Instruction:
    {instruction}

    ### Input:
    {input_text}


    ### Remember:
    Output your response in the following JSON schema format:\n{json_schema}

    ### Json Response:'''

    prompt = template.format(
          instruction=self.prompt,
          input_text=self.text,
          json_schema=self.json_schema
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": True,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "max_new_tokens": self.max_string_token_length,
            "stop": ["<|endoftext|>", "</s>"]
        }
    }
    chat = self.llm.predict(payload)
    return chat[0]["generated_text"]

  def llama(self, messages, client='huggingface', temperature=0.9, top_p=0.6, max_tokens=256):
    if client == 'huggingface':
      response = huggingface.ChatCompletion.create(
          model=self.model,
          messages=messages,
          temperature=temperature,
          top_p=top_p,
          max_tokens=max_tokens,
      )
      return response['choices'][0]['message']['content']  
    elif client == 'sagemaker':
      if self.sagemaker_model == 'hf-llm-falcon-7b-instruct-bf16-2023-08-17-23-18-56-384':
        return self.falcon()  
      prompt = build_llama2_prompt(messages)
      # hyperparameters for llm
      payload = {
        "inputs":  prompt,
        "parameters": {
          "do_sample": True,
          "top_p": top_p,
          "temperature": temperature,
          "top_k": 50,
          "max_new_tokens": max_tokens,
          "repetition_penalty": 1.03,
          "stop": ["</s>"]
        }
      }

      chat = self.llm.predict(payload)
      return chat[0]["generated_text"][len(prompt):]
    else:
      raise Exception("Sorry, only huggingface and sagemaker clients are implemented at this time.")



  def zero_shot_generate_object(self, client='huggingface'):
    template = """{prompt}\nUse the following context to inform your result json object:{context}"""

    if self.context == 'Use your imagination to populate the json schema!':
      logging.warning('You did not declare any context, the model will be inventing data...')
    if self.json_schema == {}:
      raise ValueError("Failed to find json_schema!")

    self.text = self.clean_text(self.context)
    if len(self.text) > 3800:
        self.text = self.text[:3799]

    prompt = template.format(
          prompt=self.prompt,
          context=json.dumps(self.text)
    )

    response = self.llama(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Output result in the following JSON schema format:\n{self.json_schema}\n\nJson Result:"}
        ],
        client=client,
        temperature=self.temperature,
        top_p=self.top_p,
        max_tokens=self.max_string_token_length,
    )
    print('RESULTS: ', str(response))
    start_string = self.get_json_start(response)
    start_end = self.get_json_end(start_string)
    return jsonify(json.loads(dirtyjson.loads(json.dumps(start_end))))


  ### END OF JSONFORMER LLAMA => BEGINNING OF SUMMARIZER AND BASE CHAT LLAMA ###  

  def get_last_sentence(self, text):
    if text.endswith('.'):
      return text
    
    # Find last sentence
    sentences = re.split(r'[.!?]\s*', text)
    last_sentence = sentences[-1]
    
    # Get text up until last sentence
    text = text[:text.rindex(last_sentence)]
    
    return text    

  def get_sum_response(self, name, context, client='huggingface', temperature=0.5, top_p=0.6, max_tokens=256):
      summarizer_chat_prompt = f'''You are a helpful summarizer Llama-LLM. Your job is to write a detailed description / profile of {name} based on the text provided by the user.
      In your description, you should touch on any of the following properties of {name}: its purpose, mission, programs, offerings, resources, industries, audiences, etc.
      Be eloquent and fairly succinct. This should be like a business profile or a short, informative news blurb that summarizes the text content with an emphasis on the subject, {name}.'''
      summarizer_user_prompt = f'''Write a description for {name}. Here is the source text: \n\n {context}'''  
      messages = [{'role': 'system', 'content': summarizer_chat_prompt}, {'role': 'user', 'content': summarizer_user_prompt}]
      return self.llama(messages, client, temperature, top_p, max_tokens) 

  def condense_summary(self, name, text, client='huggingface', temperature=0.5, top_p=0.6, max_tokens=150):
      summarizer_chat_prompt = f'''You are a helpful summarizer Llama-LLM. Your job is to condense / summarize whatever text is given to you. 
      Your response should be no longer than 2 sentences.'''

      user_prompt = f'''Summarize / condese the following description for {name}. Be succinct. Here is the description to shorten: {text}.'''
      messages = [{'role': 'system', 'content': summarizer_chat_prompt}, {'role': 'user', 'content': user_prompt}]
      return self.llama(messages, client, temperature, top_p, max_tokens)    


  def get_summaries(self, name, text, client='huggingface', temperature=0.5, top_p=0.6, max_tokens=256):
      context = self.clean_text(text)
      if client == 'sagemaker' and self.sagemaker_model == 'huggingface-pytorch-tgi-inference-2023-08-15-20-03-54-414':
        if len(context) > 1000:
          context = context[:999]
      else:    
        if len(context) > 15000:
          context = context[:13999]
      long_des = self.get_sum_response(name, context, client)
      short_des = self.condense_summary(name, long_des, client)
      return {'long_description': self.get_last_sentence(long_des), 'short_description': short_des} 