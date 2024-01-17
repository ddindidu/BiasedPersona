import os, json


def dir_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_toy(args):
    args.instruction_k=1
    args.qa_k = 5
    args.output_dir = os.path.join(args.output_dir, 'toy')
    return args


def save_json_file(args, persona_category, target_category, persona, instruction_no, response_list, timestamp):
    save_dir = args.output_dir
    save_dir = os.path.join(save_dir, persona_category)
    dir_checker(save_dir)

    save_file = 'p_{}_inst_{}_target_{}_{}.json'.format(persona, instruction_no, target_category, timestamp)

    save_path = os.path.join(save_dir, save_file)

    with open(save_path, encoding='utf-8', mode='w') as f:
        json.dump(response_list, f, ensure_ascii=False)

    print("FILE SAVED: {}".format(save_path))

    return