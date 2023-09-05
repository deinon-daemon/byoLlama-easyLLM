import os
import json
import boto3
import sagemaker
import functions_framework
from logger import setup_logger
from huggingface_hub import login
from easyllm.clients import huggingface
from llamachatbot import LlamaChatBot
from llamaJsonformer import LlamaJsonformer
logger = setup_logger()

def get_args(request_json, request_args):
    messages = request_json.get('messages') if request_json else None
    temperature = request_json.get('temperature', 0.9) if request_json else 0.9
    top_p = request_json.get('top_p', 0.6) if request_json else 0.6
    max_tokens = request_json.get('max_tokens', 256) if request_json else 256

    if not messages:
        messages = request_args.get('messages')
    
    return messages, temperature, top_p, max_tokens


@functions_framework.http
def llama_party(request):
    huggingface.prompt_builder = "llama2"
    session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name="us-east-1"
    )
    sess = sagemaker.Session(session)
    login(os.environ.get('HF_TOKEN'))
    huggingface.api_key=os.environ.get('HF_TOKEN') 
    request_json = request.get_json(silent=True) 
    request_args = request.args
    print('req args', request_args)
    print('req json', request_json)

    if not request_json.get('task'):
        return 'You must assign Llama a task: either, options = ["chat","summarize"] as of last update'

    if request_json.get('task') == 'summarize':
        llama = LlamaChatBot()
        if not request_json.get('name'):
            return 'You selected Llama summarizer, but did not provide the target name! include field name in req json!'

        name = request_json.get('name')   

        if not request_json.get('text'):
            return 'You selected Llama summarizer, but did not provide the target text! include field text in req json!'

        text = request_json.get('text')   

        try:
            resp = llama.get_summaries(name,text)
            return resp
        except Exception as e:
            return f'Exception thrown during summarization: {e}'

    elif request_json.get('task') == 'chat':        
        dummy_messages = [
            {"role": "system", "content": "\nYou are a proustian genius novelist! write like one in response to the user's prompt!"},
            {"role": "user", "content": "What is the sun?"},
        ]

        messages, temperature, top_p, max_tokens = get_args(request_json, request_args)

        if not messages:
            messages = dummy_messages

        try:
            # Call LLama API
            response = huggingface.ChatCompletion.create(
                model="meta-llama/Llama-2-70b-chat-hf",
                messages=messages,
                temperature=temperature, 
                top_p=top_p,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            return f'Exception thrown: {e}'


    elif request_json.get('task') == 'jsonformer':
        if not request_json.get('json_schema'):
            return 'You selected Llama jsonformer, but did not provide the target json_schema! include field json_schema in req json!'

        try:
            json_schema = json.loads(json.dumps(request_json.get('json_schema')))
        except Exception as e:
            return f'Exception {e} ... your json_schema is not valid json!'

        if not request_json.get('prompt'):
            return 'You selected Llama jsonformer, but did not provide the prompt! include field prompt in req json!'

        prompt = request_json.get('prompt')  


        if not request_json.get('context'):
            return 'You selected Llama jsonformer, but did not provide the context! include field context in req json!'

        context = request_json.get('context')

        if not request_json.get('client'):
            logger.warn("Client not set...defaulting to huggingface")
            client = 'huggingface'
        else:
            client = request_json.get('client')

        if not request_json.get('sage_model'):
            sage_model = 'hf-llm-falcon-7b-instruct-bf16-2023-08-17-23-18-56-384'
        else:
            sage_model = request_json.get('sage_model')

        if not request_json.get('model'):
            model = 'meta-llama/Llama-2-70b-chat-hf'
        else:
            model = request_json.get('model')

        messages, temperature, top_p, max_tokens = get_args(request_json, request_args)

    
        try: 
            llama = LlamaJsonformer(
                json_schema, 
                prompt, 
                context, 
                model=model,
                sagemaker_model=sage_model,
                temperature=temperature, 
                top_p=top_p, 
                max_string_token_length=max_tokens
            )
            
            response = llama.zero_shot_generate_object(client=client)

            return response
        except Exception as e:
            logger.warn(f'Error during llama build or jsonformer generation: {e}')
            return f'Error during llama build or jsonformer generation: {e}'
        
    elif request_json.get('task') == 'instruct':
        if not request_json.get('json_schema'):
            json_schema = {}
        else:
            try:
                json_schema = json.loads(json.dumps(request_json.get('json_schema')))
            except Exception as e:
                return f'Exception {e} ... your json_schema is not valid json!'

        messages, temperature, top_p, max_tokens = get_args(request_json, request_args)

        if not request_json.get('client'):
            logger.warn("Client not set...defaulting to huggingface")
            client = 'huggingface'
        else:
            client = request_json.get('client')

        if not request_json.get('sage_model'):
            sage_model = 'huggingface-pytorch-tgi-inference-2023-08-15-20-03-54-414'
        else:
            sage_model = request_json.get('sage_model')

        if not request_json.get('model'):
            model = 'meta-llama/Llama-2-70b-chat-hf'
        else:
            model = request_json.get('model')
        
        try: 
            llama = LlamaJsonformer(
                json_schema or {}, 
                model=model,
                sagemaker_model=sage_model,
                temperature=temperature, 
                top_p=top_p, 
                max_string_token_length=max_tokens
            )
           
            response = llama.llama(messages, client=client)

            return response
        except Exception as e:
            logger.warn(f'Error during llama build or instruct chat generation: {e}')
            return f'Error during llama build or instruct chat generation: {e}'

    else:
        return f"Unknown task: {request_json.get('task')}"
    