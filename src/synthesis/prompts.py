''' prompts of data synthesis for teacher model '''


TASK_POOL = {
    'hotpotqa': {
        'task_name': 'Multi-hop QA',
        'task_description': 'Multi-hop QA is a task to answer questions based on given passages. Each question usually requires reasoning based on multiple passages to answer.',
        'task_instruction': 'Answer the question based on the given passages. You may need to refer to multiple passages.',
        'task_instruction_closed_book': 'Answer the following input question.',
        'retrieval_instruction': 'Find passages that provide useful information to answer this question.',
        'task_examples': [
            {
                'input': 'Which American politician did Donahue replaced?',
                'output': 'Kelli Ward',
            }
        ],
    },
    'eli5': {
        'task_name': 'Long-form QA',
        'task_description': 'Long-form QA is a task requiring explanatory, elaborate, and in-depth answers to open-ended questions.',
        'task_instruction': 'Answer the question based on the given passages. The answer needs to be detailed, paragraph-level, and with explanations.',
        'task_instruction_closed_book': 'Answer the following input question.',
        'retrieval_instruction': 'Retrieve passages that provide a piece of good evidence for the answer.',
        'task_examples': [
            {
                'input': 'Why are the things that taste the best bad for us?',
                'output': 'Let\'s think about this from an evolutionary perspective. Way back in the day, (like way way way back) humans struggled for food just like every other animal. It was to our species evolutionary advantage to pursue food that would keep us full longer, or provide more energy than other food options. Fats, are 9 calories per gram compared to proteins and carbs that are 4 calories. Humans that preferred fats and craved them, had a higher chance of survival and passing on the fat craving trait. Fast forward to present day where food is plentiful. We are still programmed to eat high calorie foods just in case we won\'t find food for a week!',
            }
        ],
    },
    'nq': {
        'task_name': 'Single-hop QA',
        'task_description': 'Single-hop QA is a task to answer questions based on given passages. Each question usually requires reasoning based on a single passage to answer.',
        'task_instruction': 'Answer the question based on the given passages.',
        'task_instruction_closed_book': 'Answer the following input question.',
        'retrieval_instruction': 'Retrieve passages to answer the question.',
        'task_examples': [
            {
                'input': 'Who had the most wins in the nfl?',
                'output': 'Tom Brady',
            }
        ],
    },
    'fever': {
        'task_name': 'Fact Verification',
        'task_description': 'Fact Verification is a task of verifying whether a given claim is correct based on relevant passages.',
        'task_instruction': 'Verify whether the claim is correct based on the given passages. If it is correct, output "SUPPORTS", if it is wrong, output "REFUTES".',
        'task_instruction_closed_book': 'Verify whether the claim is correct. If it is correct, output "SUPPORTS", if it is wrong, output "REFUTES".',
        'retrieval_instruction': 'Retrieve passages to verify this claim.',
        'task_examples': [
            {
                'input': 'There is a movie called The Hunger Games.',
                'output': 'SUPPORTS',
            }
        ],
    },
    'wow': {
        'task_name': 'Dialogue Generation',
        'task_description': 'Dialogue Generation is a task of generating appropriate, reasonable and meaningful responses based on the previous rounds of dialogue and related information. Different speeches are separated by line breaks.',
        'task_instruction': 'Generate an appropriate, reasonable and meaningful response based on previous conversations and the following relevant passages.',
        'task_instruction_closed_book': 'Generate an appropriate, reasonable and meaningful response based on previous conversations.',
        'retrieval_instruction': 'Find passages related to the conversation topic.',
        'task_examples': [
            {
                'input': 'Ever heard of Yves Saint Laurent?\nNope, what/who are they.\nThey are a French luxury fashion house.\nOh really who founded it?',
                'output': 'Yep! It was founded by Yves Saint Laurent, believe it or not.',
            }
        ],
    },
    'trex': {
        'task_name': 'Slot Filling',
        'task_description': 'Slot Filling is a task of filling a slot, that is, given an entity and an attribute (or relationship), then filling in the specific value of the attribute. The entity and the attribute are separated by "[SEP]".',
        'task_instruction': 'Given an entity and an attribute (or relationship), fill in the specific value of the attribute based on the following passages. The entity and the attribute are separated by "[SEP]".',
        'task_instruction_closed_book': 'Given an entity and an attribute (or relationship), fill in the specific value of the attribute. The entity and the attribute are separated by "[SEP]".',
        'retrieval_instruction': 'Find passages related to the entities.',
        'task_examples': [
            {
                'input': 'Serge Blisko [SEP] occupation',
                'output': 'politician',
            }
        ],
    },
    'scifact': {
        'task_name': 'Fact Verification',
        'task_description': 'Fact Verification is a task of verifying whether a given claim is correct based on relevant passages.',
        'task_instruction': 'Verify whether the claim is correct based on the given passages. If it is correct, output "SUPPORT", if it is wrong, output "CONTRADICT".',
        'task_instruction_closed_book': 'Verify whether the claim is correct. If it is correct, output "SUPPORT", if it is wrong, output "CONTRADICT".',
        'retrieval_instruction': 'Retrieve passages to verify this claim.',
    },
    'zs-re': {
        'task_name': 'Slot Filling',
        'task_description': 'Slot Filling is a task of filling a slot, that is, given an entity and an attribute (or relationship), then filling in the specific value of the attribute. The entity and the attribute are separated by "[SEP]".',
        'task_instruction': 'Given an entity and an attribute (or relationship), fill in the specific value of the attribute based on the following passages. The entity and the attribute are separated by "[SEP]".',
        'task_instruction_closed_book': 'Given an entity and an attribute (or relationship), fill in the specific value of the attribute. The entity and the attribute are separated by "[SEP]".',
        'retrieval_instruction': 'Find passages related to the entities.',
    },
    'fiqa': {
        'task_name': 'Financial QA',
        'task_description': 'Financial QA is a task to answer questions in financial domain based on given passages.',
        'task_instruction': 'Answer the question based on the given passages.',
        'task_instruction_closed_book': 'Answer the following input question.',
        'retrieval_instruction': 'Find passages to answer the question.',
    },
    'climate-fever': {
        'task_name': 'Fact Verification',
        'task_description': 'Fact Verification is a task of verifying whether a given claim is correct based on relevant passages.',
        'task_instruction': 'Verify whether the claim is correct based on the given passages. If it is correct, output "SUPPORTS", if it is wrong, output "REFUTES", if the information is insufficient, output "NOT_ENOUGH_INFO", if the can\'t get a sufficiently confident judgment, output "DISPUTED".',
        'task_instruction_closed_book': 'Verify whether the claim is correct. If it is correct, output "SUPPORTS", if it is wrong, output "REFUTES", if the information is insufficient, output "NOT_ENOUGH_INFO", if can\'t get a sufficiently confident judgment, output "DISPUTED".',
        'retrieval_instruction': 'Retrieve passages to verify this claim.',
    },
}

QUERY_REWRITE = {
    'decomposition': '',
    'rationale': '',
}

PASSAGE_REFORM = {
    '': '',
}

crossover = 'You are a strong data generation expert. \
Below are two pieces of data, each from a different task dataset. \
The first piece of data gives the corresponding context. Consider the \
characteristics of the task and construct a new piece of data, the task of \
which is the same as the second piece of data, but you should only use context of \
the first piece of data. \'====xxx begins====\' and \'====xxx ends====\' \
indicate the beginning and end of each information respectively.\
\n\n\
====Data 1 begins====\n\
Task Instruction: {data_1_task_instruction}\n\
Input: {data_1_input}\n\
Output: {data_1_output}\n\
====Data 1 ends====\
\n\n\
====Context of Data 1 begins====\n\
{data_1_context}\n\
====Context of Data 1 ends====\
\n\n\
====Data 2 begins====\n\
Task Instruction: {data_2_task_instruction}\n\
Input: {data_2_input}\n\
Output: {data_2_output}\n\
====Data 2 ends====\
\n\n'



synthesize_one_step = 'You are a strong expert of data synthesis. \
Below, I will provide the context, the description and an example of the target task. \
Your task is to generate a piece of data for the target task based on the given context. \
The sections marked with ====xxx begins==== and ====xxx ends==== indicate the start and end of each respective part. \
Please note that the data you generate must meet the following criteria:\n\
1. Correctness, which must be logically correct and factually correct.\n\
2. Faithfulness, which must be faithful to the context.\n\
3. Quality, which must be thoughtful and sophisticated, ideally based on multiple paragraphs where applicable.\n\n\
Please note that the generated data should follow this specific format:\n\
====New data begins====\nInput:\nReference output:\n====New data ends====\n\n\
====Context begins====\n{context}\n====Context ends====\n\n\
====Target task description begins====\n{task_description}\n====Target task description ends====\n\n\
====Target task example begins====\nInput: {task_example_input}\nReference output: {task_example_output}\n====Target task example ends====\n\n\
Please ensure that your output matches the instructions above.'

synthesize_multi_step_qa = []

fact_verification = 'You need to judge the factual correctness of the input sentence based on the context. If it is correct, output VERIFIED, if it is wrong, output NOT VERIFIED.'


synthesize_with_selfmem = 'You are a strong expert of data synthesis. \
Below, I will provide the context, the description and an example of the target task. \
Your task is to generate a piece of data for the target task based on the given context. \
The sections marked with ====xxx begins==== and ====xxx ends==== indicate the start and end of each respective part. \
Please note that the data you generate must meet the following criteria:\n\
1. Correctness, which must be logically correct and factually correct.\n\
2. Faithfulness, which must be faithful to the context.\n\
3. Quality, which must be thoughtful and sophisticated, ideally based on multiple paragraphs where applicable.\n\n\
Please note that the generated data should follow this specific format:\n\
====New data begins====\nInput:\nReference output:\n====New data ends====\n\n\
Please note that, during this process, you can use your own knowledge to enrich the context.\
If you do so, please provide your own knowledge in the following format:\n\
====Own knowledge begins====\n[Describe here]\n====Own knowledge ends====\n\n\
====Context begins====\n{context}\n====Context ends====\n\n\
====Target task description begins====\n{task_description}\n====Target task description ends====\n\n\
====Target task example begins====\nInput: {task_example_input}\nReference output: {task_example_output}\n====Target task example ends====\n\n\
Please ensure that your output matches the instructions above.'

context = '[1]: single. It was backed with "Artificial Light" in Britain, and "Live Life" on the U.S. version. It peaked at #30 on the "Billboard" Hot 100 in America, the band\'s best charting American single since 1970\'s "Lola." It also charted at #30 in Canada. The track is generally cited by critics as one of the highlights from "Misfits". Stephen Thomas Erlewine of AllMusic called the track one of the two "touchstones" of the album and named it as a highlight from the album. Ken Emerson of Rolling Stone called the song "ruthless", and went on to say, The song has since\n\n\
[2]: contributing to the majority of tracks. Other band members besides Rose performed lead vocals on a number of tracks, and the albums also featured a range of guest musicians including Shannon Hoon, Michael Monroe, and Alice Cooper. Stradlin later left the band and was replaced by Gilby Clarke. In 1993 the band released ""The Spaghetti Incident?"", an album of cover versions and the band\'s first release to feature Clarke. Among others, the album featured recordings of "Raw Power" by The Stooges, "Since I Don\'t Have You" by The Skyliners and a medley of "Buick Mackane" and "Big Dumb Sex" by\n\n\
[3]: altered lyrics to "Civil War" during a concert to reference Trump. During the show of May 27 in Ireland, the band debuted their cover of "Black Hole Sun" in honor of Chris Cornell, who had died just over a week before, on May 18. Cornell had previously worked with both McKagan (in Mad Season) and Slash (appearing on the song "Promise" on "Slash"). In a 1989 interview, Rose had called Cornell the best rock vocalist at the time, and during the "Use Your Illusion tour" in the early 90s, Soundgarden opened for Guns N\' Roses in both the United States'
task_description = 'Answer the question based on the given context.'
task_example_input = 'Which Guns N\'Roses song, known for being their only number-one single in the U.S., is covered by the American alternative rock band Luna?'
task_example_output = 'Sweet Child o\' Mine'


data_checking = 'You are tasked with checking whether the following synthetic data of {task_name} task is logically correct and formatted correctly. \
The data consists of five parts: task description, example, input, output, source passages. \
The input and output of the synthetic data are based on the source passages. \
And a reasonable example of {task_name} task is provided, note that it is not based on source passages. \
Please check the following:\n\n\
1. Logical Correctness: Check whether the output correctly solves the input based on the source passages.\n\
2. Format Correctness: Check whether the input and output of the synthetic data conform to the correct format presented in the task description and the example.\n\n\
Task description: {task_description}\n\n\
Example:\nInput: {task_example_input}\nOutput: {task_example_output}\n\n\
Now, please check the following synthetic data based on source passages:\n\n\
Input: {input}\n\n\
Output: {output}\n\n\
Source passages:\n{context}\n\n\
Please note that if the above synthetic data basically meets the requirements, output "[YES]", otherwise output "[NO]".'

