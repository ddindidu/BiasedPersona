import os, json
import glob


def dir_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_toy(args):
    args.qa_k = 5
    args.output_dir = os.path.join(args.output_dir, 'toy')
    return args


def file_exist(args, persona_category, persona, instruction_no):
    save_dir = args.output_dir
    save_dir = os.path.join(save_dir, args.model, persona_category)
    save_path = os.path.join(save_dir, '*.json')

    saved_filelist = glob.glob(save_path)

    expected_filename = 'p_{}_inst_{}'.format(persona, instruction_no)

    for saved_file in saved_filelist:
        if expected_filename in saved_file:
            print("FILE EXISTS: {}".format(saved_file))
            return True

    return False



def save_json_file(args, persona_category, target_category, persona, instruction_no, response_list, timestamp):
    save_dir = args.output_dir
    save_dir = os.path.join(save_dir, args.model, persona_category)
    dir_checker(save_dir)

    save_file = 'p_{}_inst_{}_target_{}_{}.json'.format(persona, instruction_no, target_category, timestamp)

    save_path = os.path.join(save_dir, save_file)

    with open(save_path, encoding='utf-8', mode='w') as f:
        json.dump(response_list, f, ensure_ascii=False)

    print("FILE SAVED: {}".format(save_path))

    return