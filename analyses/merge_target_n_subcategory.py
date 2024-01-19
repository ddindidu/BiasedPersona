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
    score_df = call_score_df(args, args.persona_category, args.target_category)
    persona_df = call_persona_df(args.source_dir, args.persona_file, args.persona_category)
    print(score_df)
    print(persona_df)
    merged_df = persona_df.merge(score_df, how='inner', on='Name')
    print(merged_df)

    dir_path = args.dir_path
    file_name = 'merged_'+ args.file_name.format(args.persona_category, args.target_category)
    file_path = os.path.join(dir_path, file_name)
    merged_df.to_csv(file_path, index=False)

    return


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_path', type=str, default='./PersonaTargetScore')
    parser.add_argument('--file_name', type=str, default='{}2{}.csv')
    parser.add_argument('--source_dir', type=str, default='./../source/Persona')
    parser.add_argument('--persona_file', type=str, default='persona_list.csv')

    parser.add_argument('--persona_category', type=str, default='Baseline')
    parser.add_argument('--target_category', type=str, default='SES')

    parser.add_argument('--target_level', type=str, default='subcategory')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for p in ['Baseline', args.target_category]:
        args.persona_category = p
        main(args)