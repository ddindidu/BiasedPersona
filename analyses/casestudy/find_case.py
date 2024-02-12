import os, glob, argparse
import json
import pandas as pd

def get_persona_list(persona_category):
    if persona_category == 'Race_ethnicity':
        persona_list = ['Asian', 'Arab', 'Black', 'Caucasian', 'White']
    elif persona_category == 'Age':
        persona_list = ['kid', 'elder']
    elif persona_category == 'Religion':
        persona_list = ['Christian', 'Protestant', 'Hindu', 'Muslim', 'Buddhist']
    return persona_list

def all_equal(l):
    len_set = len(set(l))
    if len_set <= 1:
        return True
    else:
        return False


def main(args):
    category = args.category

    dir_path = os.path.join(args.result_dir, args.model, category)
    file_list = glob.glob(os.path.join(dir_path, '*.json'))
    file_list.sort()
    #print(file_list)

    persona_list = get_persona_list(category)
    persona_list.sort()

    dfs = ()
    for f_name in file_list:
        with open(f_name, 'r') as f:
            data = json.load(f)
            f.close()
        dfs += (data, )

    def return_question(d):
        return d['context'], d['question_polarity'], d['question']
    def return_elements(d):
        res = d['response']
        return res['origin'], res['refined']

    text_file = open('./{}.txt'.format(category), 'w')

    my_text = ""

    n = 0
    for i in range(len(dfs[0])):
        if dfs[0][i]['context_condition'] == 'disambig':
            continue
        origin_responses, refined_responses = [], []

        context, q_pol, question = return_question(dfs[0][i])

        for n_persona in range(len(dfs)):
            origin, refined = return_elements(dfs[n_persona][i])
            origin_responses.append(origin)
            refined_responses.append(refined)

        if not all_equal(refined_responses):
            n+=1
            print("Q: {}".format(question))
            my_text += "C: {}\n".format(context)
            my_text += "Q: ({}) {}\n".format(q_pol, question)

            for idx, p in enumerate(persona_list):
                print("P: {} / {}".format(p, origin_responses[idx]))
                my_text += "P:{} / {}\n".format(p, origin_responses[idx])
            print('--------------------')
            my_text += "==================================================\n"

    text_file.write(my_text)
    text_file.close()
    print(n)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, default='./../../results/reason/refined')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')

    parser.add_argument('--category', type=str, default='Age')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    main(args)