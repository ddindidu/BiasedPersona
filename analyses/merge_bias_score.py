import os, glob
import argparse
import pandas as pd
from math import isnan


def dir_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def call_dfs(args, persona, category, rp, cc):
    result_dir = os.path.join(args.result_dir, args.model)

    dir_baseline = os.path.join(result_dir, persona)
    #dir_category = os.path.join(result_dir, category)

    if args.model != 'gpt-3.5-turbo-0613':
        overall_f = 'inst_0_*2{}_overall_score.csv'.format(category)
        ambig_f = 'inst_0_*2{}_ambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        disambig_f = 'inst_0_*2{}_disambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
    else:
        overall_f = 'aver_*2{}_overall_score.csv'.format(category)
        ambig_f = 'aver_*2{}_ambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        disambig_f = 'aver_*2{}_disambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)

    def call_files(dir_path, overall_f, ambig_f, disambig_f):
        overall_file_path = glob.glob(os.path.join(dir_path, overall_f))[0]
        ambig_file_path = glob.glob(os.path.join(dir_path, ambig_f))[0]
        disambig_file_path = glob.glob(os.path.join(dir_path, disambig_f))[0]

        df_overall = pd.read_csv(overall_file_path, index_col=0)
        df_ambig = pd.read_csv(ambig_file_path, index_col=0)
        df_disambig = pd.read_csv(disambig_file_path, index_col=0)

        return df_overall, df_ambig, df_disambig


    df_overall_base, df_ambig_base, df_disambig_base = call_files(dir_baseline, overall_f, ambig_f, disambig_f)
    #df_overall_persona, df_ambig_persona, df_disambig_persona = call_files(dir_category, overall_f, ambig_f, disambig_f)

    return df_overall_base, df_ambig_base, df_disambig_base


def calcul_bias_target_n_persona(df):
    columns = df.columns
    # calculate target_bias
    target_bias_list = []
    for idx, row in df.iterrows():
        target_bias = 0
        for col in columns:
            value = row[col]
            if isnan(value):
                value = 0
            target_bias += abs(value)
            #print(value, end=' ')
        #print("Target Bias: ", target_bias)
        target_bias_list.append(target_bias)

    # calculate persona_bias
    persona_bias_list = []
    df_baseline = df.loc['Baseline']
    for idx, row in df.iterrows():
        persona_bias = 0
        for col in columns:
            b_t = df_baseline[col]
            p_t = row[col]
            if isnan(b_t):
                b_t = 0
            if isnan(p_t):
                p_t = 0
            persona_bias += abs(p_t - b_t)
        persona_bias_list.append(persona_bias)

    df['TB'] = target_bias_list
    df['PB'] = persona_bias_list
    print(df)
    return df


def refine_column_names(df, context_condition):
    columns = df.columns

    new_columns = []
    for col in columns:
        new_name = "{}_{}".format(col, context_condition)
        new_columns.append(new_name)

    df.columns = new_columns

    return df


def main(args):
    category = args.category
    rp, cc = args.rp, args.cc

    df_overall_base, df_ambig_base, df_disambig_base = call_dfs(args, 'Baseline', category, rp, cc)
    df_overall_persona, df_ambig_persona, df_disambig_persona = call_dfs(args, category, category, rp, cc)

    df_overall = pd.concat([df_overall_base, df_overall_persona], axis=0)
    df_ambig = pd.concat([df_ambig_base, df_ambig_persona], axis=0)
    df_disambig = pd.concat([df_disambig_base, df_disambig_persona], axis=0)

    df_ambig_calcul = calcul_bias_target_n_persona(df_ambig)
    df_disambig_calcul = calcul_bias_target_n_persona(df_disambig)

    df_ambig_calcul = refine_column_names(df_ambig_calcul, 'ambig')
    df_disambig_calcul = refine_column_names(df_disambig_calcul, 'disambig')

    df_merged = pd.concat([df_overall, df_ambig_calcul, df_disambig_calcul], axis=1)

    dir_save = os.path.join(args.save_dir, args.result_dir, args.model, args.category)
    dir_checker(dir_save)
    file_name = 'merged_total_rp_{}_cc_{}.csv'.format(rp, cc)
    save_path = os.path.join(dir_save, file_name)

    df_merged.to_csv(save_path)
    print("FILE SAVED: {}".format(save_path))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, default='Bias_Score_newDeno')
    parser.add_argument('--save_dir', type=str, default='total_merged')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--model', type=str, default='meta-llama/Llama-2-70b-chat-hf')

    parser.add_argument('--category', type=str, default='SES')

    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    #print(args)

    points = [(2,1), (1,1), (1,0)]
    for point in points:
        args.rp = point[0]
        args.cc = point[1]
        print(args)
        main(args)
