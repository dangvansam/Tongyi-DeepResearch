import time
from typing import List, Iterator
import copy
import json
import logging
import random
import os
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Literal, Union

import openai
from openai import OpenAIError, RateLimitError 

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel, simulate_response_completion_with_chat
from qwen_agent.llm.schema import ASSISTANT, Message, FunctionCall
from qwen_agent.log import logger
import datetime
def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "parse_file", "description": "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3.", "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}, "description": "The file name of the user uploaded local files to be parsed."}}, "required": ["files"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

"""

@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
            
        try:
            MAX_RETRIES = 5 
            INITIAL_DELAY = 2  
            CONTENT_THRESHOLD = 50
            REASONING_THRESHOLD = 50
            response = None 
        
            for attempt in range(MAX_RETRIES):
                try:
                    response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
                    break

                except RateLimitError as ex:
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"API rate limit error after {MAX_RETRIES} retries. Raising exception.")
                        raise ModelServiceError(exception=ex) from ex

                    delay = INITIAL_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)

                except OpenAIError as ex:
                    logger.error(f"An OpenAI error occurred: {ex}")
                    raise ModelServiceError(exception=ex) from ex
            # response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                            yield [
                                Message(
                                    role=ASSISTANT,
                                    content='',
                                    reasoning_content=choice.delta.reasoning_content
                                )
                            ]
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            yield [Message(role=ASSISTANT, content=choice.delta.content, reasoning_content='')]
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            function_name = choice.delta.tool_calls[0].function.name
                            function_call = {
                                'name': function_name,
                                'arguments': json.loads(choice.delta.tool_calls[0].function.arguments)
                            }
                            function_json = json.dumps(function_call, ensure_ascii=False)
                            yield [Message(role=ASSISTANT, content=f'<tool_call>{function_json}</tool_call>')]
                    logger.info(f'delta_stream message chunk: {chunk}')
            else:

                full_response = ''
                full_reasoning_content = ''
                content_buffer = ''
                reasoning_content_buffer = ''
    
                for chunk in response:
                    if not chunk.choices:
                        continue
                    
                    choice = chunk.choices[0]
                    new_content = choice.delta.content if hasattr(choice.delta, 'content') and choice.delta.content else ''
                    new_reasoning = choice.delta.reasoning if hasattr(choice.delta, 'reasoning') and choice.delta.reasoning else ''
                    has_tool_calls = hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls
    
                    if new_reasoning:
                        full_reasoning_content += new_reasoning
                        reasoning_content_buffer += new_reasoning
                    
                    if new_content:
                        full_response += new_content
                        content_buffer += new_content
                    
                    if has_tool_calls:
                        function_name = choice.delta.tool_calls[0].function.name
                        function_call = {
                            'name': function_name,
                            'arguments': json.loads(choice.delta.tool_calls[0].function.arguments)
                        }
                        function_json = json.dumps(function_call, ensure_ascii=False)
                        logger.info(json.dumps(function_call, ensure_ascii=False, indent=4))
                        full_response += f'<tool_call>{function_json}</tool_call>'
                        content_buffer += '<tool_call>' 
    
                    if (len(content_buffer) >= CONTENT_THRESHOLD or
                        len(reasoning_content_buffer) >= REASONING_THRESHOLD or
                        '\n' in new_content or
                        '\n' in new_reasoning):
                        
                        yield [Message(role=ASSISTANT, content=full_response, reasoning_content=full_reasoning_content)]
    
                        content_buffer = ''
                        reasoning_content_buffer = ''
    
                    logger.info(f'message chunk: {chunk}')
    
                if content_buffer or reasoning_content_buffer:
                    yield [Message(role=ASSISTANT, content=full_response, reasoning_content=full_reasoning_content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            if hasattr(response.choices[0].message, 'reasoning_content'):
                return [
                    Message(role=ASSISTANT,
                            content=response.choices[0].message.content,
                            reasoning_content=response.choices[0].message.reasoning_content)
                ]
            else:
                return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        # if delta_stream:
        #     raise NotImplementedError('Please use stream=True with delta_stream=False, because delta_stream=True'
        #                               ' is not implemented for function calling due to some technical reasons.')
        generate_cfg = copy.deepcopy(generate_cfg)
        for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
            if k in generate_cfg:
                del generate_cfg[k]
        messages = simulate_response_completion_with_chat(messages)
        return self._chat(messages, stream=stream, delta_stream=delta_stream, generate_cfg=generate_cfg)

    def _chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful

        messages = [msg.model_dump() for msg in messages]
        return_messages = []
        messages[0]["content"] = SYSTEM_PROMPT + "Current date: " + str(today_date())
        for i in messages:
            i["content"] = i["content"].replace("<think>\n<think>\n","<think>\n\n")
            return_messages.append(i)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return return_messages
