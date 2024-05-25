import os, datetime, argparse

from persona import call_persona_list, call_persona_instruction
from qa import call_bbq
from generate import get_generative_model, generate
from utils import set_toy, file_exist, save_json_file


def test(args, persona_instruction, persona_category, persona, qa_dataset, client):
    pa_instruction = ""
    if persona_category != 'Baseline':
        pa_instruction = persona_instruction.format(persona)

    response_dataset = []

    for qa_item in qa_dataset:
        #print(qa_item)
        context = qa_item['context']
        question = qa_item['question']
        options = []
        for idx in range(3):
            options.append(qa_item['ans{}'.format(idx)])
        #print(question, options)

        response = generate(args, client, pa_instruction, context, question, options)

        #print(response)

        qa_item['response'] = {
            'origin': response
        }
        response_dataset.append(qa_item)

    #print(response_dataset)

    return response_dataset


def main(args):
    g_model = get_generative_model(args)

    if args.toy == 1:
        args = set_toy(args)

    persona_instruction = call_persona_instruction(n=args.instruction_end)

    persona_category = args.persona_category
    target_category = args.target_category

    persona_dict = call_persona_list(args.source_dir, args.persona_file, persona_category)
    persona_list = persona_dict['persona_list']
    if args.reasoning == 1:
        if args.persona_category == 'Age':
            persona_list = ['kid', 'elder']
        if args.persona_category == 'Race_ethnicity':
            persona_list = ['Caucasian', 'White', 'Black', 'Arab', 'Asian']
        if args.persona_category == 'Religion':
            persona_list = ['Christian', 'Protestant', 'Hindu', 'Muslim', 'Buddhist']

    qa_dataset = call_bbq(args.source_dir, target_category, args.qa_k)

    if args.toy == 1:
        print(persona_instruction)
        print(persona_list)
        print(qa_dataset)

    for p_no, p in enumerate(persona_list):
        for inst_no, inst in enumerate(persona_instruction):
            if inst_no < args.instruction_start:
                continue
            if file_exist(args, persona_category, p, inst_no):
                continue
            print(p_no, p, inst_no, inst)
            timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")  # identifiable_token
            response_list = test(args, inst, persona_category, p, qa_dataset, g_model)
            save_json_file(args, persona_category, target_category, p, inst_no, response_list, timestamp)

    return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./../source')
    parser.add_argument('--output_dir', type=str, default='./../results/origin')
    parser.add_argument('--persona_file', type=str, default='persona_list.csv')

    parser.add_argument('--reasoning', type=int, default=0)

    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    #parser.add_argument('--temperature', type=int, default=0)
    parser.add_argument('--api_key', type=int, default=0)
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--deepinfra_api_key', type=str, default='5r7osFo18xFNBwwTDVb4IOWR22PZ4S9F')
    parser.add_argument('--google_ai_api_key', type=str, default="AIzaSyC86w89PjZpPhgkNGo3KQsb5c-b0awIJnQ")

    parser.add_argument('--persona_category', type=str, default='SES')
    parser.add_argument('--target_category', type=str, default='SES')

    parser.add_argument('--instruction_start', type=int, default=0)
    parser.add_argument('--instruction_end', type=int, default=3)
    parser.add_argument('--qa_k', type=int, default=None)

    parser.add_argument('--toy', type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)