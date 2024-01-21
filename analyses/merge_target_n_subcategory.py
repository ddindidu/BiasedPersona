import os, argparse
import pandas as pd

import sys
sys.path.append('./../test/')
from persona import call_persona_df

def call_score_df(args, persona_category, target_category):
    dir_path = args.dir_path
    file_name = args.file_name.format(persona_category, target_category)
    file_path = os.path.join(dir_path, file_name)

    score_df = pd.read_csv(file_path, index_col=[0])
    score_df.reset_index(inplace=True)
    score_df = score_df.rename(columns={'index': 'Name'})

    return score_df




def main(args):
    score_df = call_score_df(args, args.persona_category, args.target_category) # (persona_n*target_n)
    persona_df = call_persona_df(args.source_dir, args.persona_file, args.persona_category)   # (persona_n, 3)
    print(score_df)
    print(persona_df)
    merged_df = persona_df.merge(score_df, how='inner', on='Name')  # (persona_n*(3+target_n))
    if (args.persona_category == 'Nationality') and (args.target_category == 'Nationality') and (args.target_level == 'Name'):
        name_list = merged_df['Name'].tolist()  # persona order list
        print(name_list)
        column_list = list(merged_df)
        column_list = column_list[:3]+name_list
        merged_df = merged_df[column_list]

    print(merged_df)

    dir_path = args.dir_path
    file_name = 'merged_'+ args.file_name.format(args.persona_category, args.target_category)
    file_path = os.path.join(dir_path, file_name)
    merged_df.to_csv(file_path, index=False)
    print(file_path)

    return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_path', type=str, default='./PersonaTargetScore')
    parser.add_argument('--file_name', type=str, default='{}2{}.csv')
    parser.add_argument('--source_dir', type=str, default='./../source/Persona')
    parser.add_argument('--persona_file', type=str, default='persona_list.csv')

    parser.add_argument('--persona_category', type=str, default='Baseline')
    parser.add_argument('--target_category', type=str, default='Nationality')

    parser.add_argument('--target_level', type=str, default='Name')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for p in ['Baseline', args.target_category]:
        args.persona_category = p
        if args.target_level == 'Name':
            args.file_name = '{}2{}_name.csv'
        main(args)