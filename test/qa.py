import os
import json


def call_bbq(source_dir, category, n=None):
    dir_path = os.path.join(source_dir, 'BBQ/data')
    file_name = "{}.jsonl".format(category)
    file_path = os.path.join(dir_path, file_name)

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            dict = json.loads(line)
            data.append(dict)

    if n is not None:
        data = data[:n]

    return data


if __name__ == "__main__":
    source_dir = './../source'
    category = "SES"

    res = call_bbq(source_dir, category, n=5)
    print(res)
    print(len(res))
    print(res[0]['context_condition'])

    res = call_bbq(source_dir, category, n=None)
    print(res)
    print(len(res))
    print(res[0]['context_condition'])