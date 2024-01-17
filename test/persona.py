import os
import pandas as pd



source_dir = "./../source"


def call_persona_instruction(n = None):
    file_name = 'persona_instruction.txt'
    persona_dir = 'Persona'
    file_path = os.path.join(source_dir, persona_dir, file_name)

    inst = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line= line.strip()
            inst.append(line)

    if n is not None:
        inst = inst[:n]

    return inst


def call_persona_list(source_dir, file_name, category):
    #   PARAM
    #
    #   RETURN
    #   {
    #       'generic_category':  # str
    #       'specific_category': # str
    #       'persona_list':     # list of str
    #   }
    file_name = file_name
    persona_dir = 'Persona'
    file_path = os.path.join(source_dir, persona_dir, file_name)
    persona_df = pd.read_csv(file_path, header=0)

    selected_persona_df = persona_df[(persona_df['Category']==category)]

    persona_list = selected_persona_df['Name'].tolist()
    subcategory = selected_persona_df['Subcategory'].tolist()[0]
    if type(subcategory) is not str:
        subcategory = None

    return {
        'Category': category,
        'Subcategory': subcategory,
        'persona_list': persona_list,
    }


if __name__ == "__main__":
    res = call_persona_list(source_dir, 'persona_list.csv', category='Baseline')
    print(res)
    res = call_persona_list(source_dir, 'persona_list.csv', category='Race_ethnicity')
    print(res)
    res = call_persona_list(source_dir, 'persona_list.csv', category='SES')
    print(res)