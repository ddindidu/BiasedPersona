import os, argparse
import glob
import json

import sys
sys.path.append('./../../test/')
from persona import call_persona_list
from utils import dir_checker


def save_list2json(args, f_name, list):
    f_name = os.path.basename(f_name)

    output_dir = os.path.join(args.model, args.persona_category)
    dir_checker(output_dir)
    output_file = '{}_{}'.format('refined', f_name)
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, encoding='utf-8', mode='w') as f:
        json.dump(list, f)
        f.close()
    print("FILE SAVED: {}".format(output_path))


def process_response(x, options):
    for idx in range(3):
        option = options[idx]
        if option.lower() in x.lower():
            return idx

    option_format = ['({})', '{})', '{}']
    alphabet_options = ['A', 'B', 'C']
    for idx, alpha_opt in enumerate(alphabet_options):
        for form in option_format:
            option = form.format(alpha_opt)
            if option in x:
                return idx

    return 9


def main(args, persona_category, target_category):
    persona_dict = call_persona_list(args.source_dir, 'persona_list.csv', persona_category)
    persona_list = persona_dict['persona_list']

    result_dir = os.path.join(args.result_dir, args.model)
    result_dir = os.path.join(result_dir, persona_category)
    file_name = '*{}*.json'.format(target_category)

    file_list = glob.glob(os.path.join(result_dir, file_name))
    print(file_list)

    persona_list.sort()
    file_list.sort()

    for p_no, persona_name in enumerate(persona_list):
        for inst_no in range(args.instruction_k):
            file_idx = p_no*args.instruction_k + inst_no
            f_name = file_list[file_idx]
            with open(f_name, 'r') as f:
                origin_data = json.load(f)
                f.close()
            # print(origin_data)

            refined_data = []
            for item in origin_data:
                origin_response = item['response']['origin']
                options = []
                for i in range(3):
                    options.append(item['ans{}'.format(i)])
                refined_response = process_response(origin_response, options)
                #print(refined_response)

                item['response']['refined'] = refined_response
                refined_data.append(item)

            save_list2json(args, f_name, refined_data)
    return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./../../source')
    parser.add_argument('--result_dir', type=str, default='./../origin')

    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-70b-chat-hf')

    parser.add_argument('--persona_category', type=str, default='Race_ethnicity')
    parser.add_argument('--target_category', type=str, default='Race_ethnicity')

    parser.add_argument('--instruction_k', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    persona_category = args.persona_category
    target_category = args.target_category

    main(args, persona_category, target_category)

