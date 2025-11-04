''' synthesis or enhance data by teacher model '''

from typing import List, Dict
from prompts import (
    synthesize_one_step,
    data_checking,
)
import re
import openai
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def call_teacher_model(client: openai.OpenAI,
                       messages: List[Dict],
                       model_name: str,
                       temperature: float = 0):
    '''
    prompt teacher model.
    '''
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )

        return completion.choices[0].message.content
    
    except Exception as e:
        logger.error(f'Exception occurred during calling {model_name}: {e}')

        return None


def parse_teacher_model_output(output: str,
                               new_context: bool = False):
    '''
    parse new data from teacher model's output.
    - new_context: if true, there maybe new context in the output
    '''
    
    output = output.replace('\n', '')

    pattern_1 = r'====New data begins====(.*?)====New data ends===='
    match_1 = re.search(pattern_1, output)
    if not match_1:
        logger.warning(f'The output of teacher model does not follow the output format instruction: {output}')
        return None
    new_data = match_1.group(1)

    pattern_2 = r'Input:\s*(.*?)\s*Reference output:'
    match_2 = re.search(pattern_2, new_data)
    if not match_2:
        logger.warning(f'The output of teacher model does not follow the output format instruction: {output}')
        return None
    new_data_input = match_2.group(1)

    pattern_3 = r'Reference output:\s*(.*)'
    match_3 = re.search(pattern_3, new_data)
    if not match_3:
        logger.warning(f'The output of teacher model does not follow the output format instruction: {output}')
        return None
    new_data_output = match_3.group(1)

    if new_context:
        pattern_4 = r'====Own knowledge begins====(.*?)====Own knowledge ends===='
        match_4 = re.search(pattern_4, output)
        if match_4:
            return (new_data_input, new_data_output, match_4.group(1))
        else:
            return (new_data_input, new_data_output, None)

    return (new_data_input, new_data_output)


def get_prompt_for_new_data_one_step(context: str,
                                     task_name: str,
                                     task_description: str,
                                     task_example_input: str,
                                     task_example_output: str,
                                     client: openai.OpenAI,
                                     model_name: str,
                                     temperature: float = 0.5):
    '''
    synthesize new data with only one step for all tasks.
    '''

    prompt = synthesize_one_step.format(
        context=context,
        task_description=task_name + ': ' + task_description,
        task_example_input=task_example_input,
        task_example_output=task_example_output
    )
    messages = [
        {'role': 'system', 'content': 'You are ChatGPT, a large language model trained by OpenAI.'},
        {'role': 'user', 'content': prompt}
    ]
    output = call_teacher_model(client=client,
                                messages=messages,
                                model_name=model_name,
                                temperature=temperature)
    if output is None:
        logger.warning('Synthesis failed')
        return None
    
    return output


def data_checking_filetering(context: str,
                             task_name: str,
                             task_description: str,
                             task_example_input: str,
                             task_example_output: str,
                             client: openai.OpenAI,
                             input: str,
                             output: str,
                             model_name: str,
                             temperature: float = 0.5):
    prompt = data_checking.format(
        task_name=task_name,
        task_description=task_description,
        task_example_input=task_example_input,
        task_example_output=task_example_output,
        input=input,
        output=output,
        context=context,
    )
    messages = [
        {'role': 'system', 'content': 'You are ChatGPT, a large language model trained by OpenAI.'},
        {'role': 'user', 'content': prompt}
    ]
    output = call_teacher_model(client=client,
                                messages=messages,
                                model_name=model_name,
                                temperature=temperature)
    if '[YES]' in output:
        return True
    else:
        return False


if __name__ == '__main__':
    output = '====New data begins==== Input: What was the title of the album released by Guns N\' Roses in 1993, which featured cover versions of various songs? Reference output: The Spaghetti Incident? ====New data ends===='
    new_data = parse_teacher_model_output(output)
    print(new_data)
