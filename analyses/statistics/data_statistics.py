import os, argparse
import json


def get_target_list(category):
    targets = []
    if category == "Age":
        targets = ['nonOld', 'old']
    if category == "Race_ethnicity":
        targets = ['African', 'African American', 'Black', 'Caucasian', 'White', 'European', 'Roma', 'Native American',
                   'South American', 'Hispanic', 'Latino', 'Jewish', 'Arab', 'Middle Eastern', 'Asian']
    if category == "Religion":
        targets = ["Atheist", "Christian", "Protestant", "Catholic", "Mormon", "Jewish", "Muslim", "Sikh", "Hindu",
                   "Buddhist"]
    if category == "SES":
        targets = ['highSES', 'lowSES']
    if category == "Sexual_orientation":
        targets = ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual']
    return targets


def call_bbq(source_dir, target_category, n=None):
    dir_path = os.path.join(source_dir)
    file_name = "{}.jsonl".format(target_category)
    file_path = os.path.join(dir_path, file_name)

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            dict = json.loads(line)
            data.append(dict)

    if n is not None:
        data = data[:n]

    return data


def count_target_options(data):
    target_dict = {'ambig': 0, 'disambig': 0}
    for d in data:
        answer_info = d['answer_info']
        target_dict[d['context_condition']] += 1

        for i in range(3):
            target = answer_info['ans{}'.format(i)][1]
            if 'F-' in target or 'M-' in target:
                target = target[2:]
            if target != 'unknown':
                key = '{}_{}'.format(d['context_condition'], target)

                if key in target_dict:
                    target_dict[key] += 1
                else:
                    target_dict[key] = 1

    return target_dict

def main(args):
    domains = args.domains

    for domain in domains:
        data = call_bbq(args.source_dir, domain)
        target_dict = count_target_options(data)
        print("DOMAIN: {}\tn(Q): {}".format(domain, len(data)))
        print(dict(sorted(target_dict.items())))



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./../../source/BBQ/data')
    parser.add_argument('--source_file', type=str, default='{}.jsonl')  # domain

    parser.add_argument('--domains', type=list, default=['Age', 'Religion', 'Race_ethnicity', 'SES', 'Sexual_orientation'])


    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)