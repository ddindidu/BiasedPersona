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
        ambig_abs_f = 'inst_0_*2{}_ambig_abs_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        disambig_f = 'inst_0_*2{}_disambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        disambig_abs_f = 'inst_0_*2{}_disambig_abs_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
    else:
        overall_f = 'aver_*2{}_overall_score.csv'.format(category)
        ambig_f = 'aver_*2{}_ambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        ambig_abs_f = 'aver_*2{}_ambig_abs_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        disambig_f = 'aver_*2{}_disambig_score_rp_{}_cc_{}.csv'.format(category, rp, cc)
        disambig_abs_f = 'aver_*2{}_disambig_abs_score_rp_{}_cc_{}.csv'.format(category, rp, cc)

    def call_files(dir_path, overall_f, ambig_f, ambig_abs_f, disambig_f, disambig_abs_f):
        overall_file_path = glob.glob(os.path.join(dir_path, overall_f))[0]
        ambig_file_path = glob.glob(os.path.join(dir_path, ambig_f))[0]
        ambig_abs_file_path = glob.glob(os.path.join(dir_path, ambig_abs_f))[0]
        disambig_file_path = glob.glob(os.path.join(dir_path, disambig_f))[0]
        disambig_abs_file_path = glob.glob(os.path.join(dir_path, disambig_abs_f))[0]

        df_overall = pd.read_csv(overall_file_path, index_col=0)
        df_ambig = pd.read_csv(ambig_file_path, index_col=0)
        df_ambig_abs = pd.read_csv(ambig_abs_file_path, index_col=0)
        df_disambig = pd.read_csv(disambig_file_path, index_col=0)
        df_disambig_abs = pd.read_csv(disambig_abs_file_path, index_col=0)

        return df_overall, df_ambig, df_ambig_abs, df_disambig, df_disambig_abs

    df_overall, df_ambig, df_ambig_abs, df_disambig, df_disambig_abs = call_files(dir_baseline, overall_f, ambig_f, ambig_abs_f, disambig_f, disambig_abs_f)
    df_overall.fillna(0, inplace=True)
    df_ambig.fillna(0, inplace=True)
    df_ambig_abs.fillna(0, inplace=True)
    df_disambig.fillna(0, inplace=True)
    df_disambig_abs.fillna(0, inplace=True)
    return df_overall, df_ambig, df_ambig_abs, df_disambig, df_disambig_abs


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

    target_n = len(df.columns)
    df['TB'] = [x/target_n for x in target_bias_list]
    df['PB'] = [x/target_n for x in persona_bias_list]

    # TB0-TB_p
    '''
    diff_tb = []
    TB_baseline = df.at['Baseline', 'TB']
    for idx, row in df.iterrows():
        TB_p = row['TB']
        diff = TB_p - TB_baseline
        diff_tb.append(diff)
    df['diff_TB'] = diff_tb
    '''
    print(df)
    return df


def calcul_overall_diff(df):
    bs_a_base = df.at['Baseline', 'BS_a']
    bs_d_base = df.at['Baseline', 'BS_d']

    diff_bs_a, diff_bs_d = [], []
    for idx, row in df.iterrows():
        diff_bs_a_p = row['BS_a'] - bs_a_base
        diff_bs_d_p = row['BS_d'] - bs_d_base
        diff_bs_a.append(diff_bs_a_p)
        diff_bs_d.append(diff_bs_d_p)
    df['Diff_BS_a'] = diff_bs_a
    df['Diff_BS_d'] = diff_bs_d
    return df


def refine_column_names(df, context_condition):
    columns = df.columns

    new_columns = []
    for col in columns:
        new_name = "{}_{}".format(col, context_condition)
        new_columns.append(new_name)

    df.columns = new_columns

    return df


def drop_invalid_identity(df):
    labels = ["Alaskan", "Orthodox", ]

    persona = df.index
    target = df.columns

    for lab in labels:
        if lab in persona:
            df = df.drop(labels=lab, axis=0)
        if lab in target:
            df = df.drop(labels=lab, axis=1)
    return df


def sort_persona_names(df_overall, df_amb, df_amb_abs, df_dis, df_dis_abs, category):
    if category == 'Age':
        sorter = ['boy', 'girl', 'kid', 'man', 'woman', 'elder']
    elif category == 'Race_ethnicity':
        sorter = ['African', 'African American', 'Black',
                  'Caucasian', 'White', 'European', 'Roma',
                  'Native American', 'South American', 'Hispanic', 'Latino',
                  'Jewish',
                  'Arab', 'Middle Eastern',
                  'Asian']
    elif category == 'Religion':
        sorter = ['Atheist', 'Christian', 'Protestant', 'Catholic', 'Mormon', 'Jewish', 'Muslim', 'Sikh', 'Hindu', 'Buddhist']
    elif category == 'SES':
        sorter = ['chemical engineer', 'chemist', 'dentist', 'judge', 'lawyer', 'optometrist', 'pharmacist', 'physician', 'physics teacher', 'professor', 'psychologist', 'scientist',
                  'bartender', 'cashier', 'construction worker', 'farm worker', 'janitor', 'line cook', 'parking attendant', 'receptionist', 'sales clerk', 'server', 'taxi driver', 'truck driver',]
    elif category == 'Sexual_orientation':
        sorter = ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual']

    row = ['Baseline']+sorter
    column = sorter

    # resort index
    df_overall = df_overall.reindex(row)
    df_amb = df_amb.reindex(row)
    df_amb_abs = df_amb_abs.reindex(row)
    df_dis = df_dis.reindex(row)
    df_dis_abs = df_dis_abs.reindex(row)

    # resort column
    score_sorter = ['BS_a', 'BS_d', 'Diff_Bias_a', 'Diff_Bias_d', 'Acc_a', 'Acc_d']
    df_overall = df_overall.reindex(score_sorter, axis=1)
    if category in ['Race_ethnicity', 'Religion', 'Sexual_orientation']:
        df_amb = df_amb.reindex(column, axis=1)
        df_amb_abs = df_amb_abs.reindex(column, axis=1)
        df_dis = df_dis.reindex(column, axis=1)
        df_dis_abs = df_dis_abs.reindex(column, axis=1)

    return df_overall, df_amb, df_amb_abs, df_dis, df_dis_abs


def main(args):
    category = args.category
    rp, cc = args.rp, args.cc

    df_overall_base, df_ambig_base, df_ambig_abs_base, df_disambig_base, df_disambig_abs_base = call_dfs(args, 'Baseline', category, rp, cc)
    df_overall_persona, df_ambig_persona, df_ambig_abs_persona, df_disambig_persona, df_disambig_abs_persona = call_dfs(args, category, category, rp, cc)

    df_overall = pd.concat([df_overall_base, df_overall_persona], axis=0)
    df_ambig = pd.concat([df_ambig_base, df_ambig_persona], axis=0)
    df_ambig_abs = pd.concat([df_ambig_abs_base, df_ambig_abs_persona], axis=0)
    df_disambig = pd.concat([df_disambig_base, df_disambig_persona], axis=0)
    df_disambig_abs = pd.concat([df_disambig_abs_base, df_disambig_abs_persona], axis=0)

    df_overall = drop_invalid_identity(df_overall)
    df_ambig = drop_invalid_identity(df_ambig)
    df_ambig_abs = drop_invalid_identity(df_ambig_abs)
    df_disambig = drop_invalid_identity(df_disambig)
    df_disambig_abs = drop_invalid_identity(df_disambig_abs)

    df_overall, df_ambig, df_ambig_abs, df_disambig, df_disambig_abs = sort_persona_names(df_overall, df_ambig, df_ambig_abs, df_disambig, df_disambig_abs, category)

    df_overall_calcul = calcul_overall_diff(df_overall)

    df_ambig_calcul = calcul_bias_target_n_persona(df_ambig)
    df_ambig_abs_calcul = calcul_bias_target_n_persona(df_ambig_abs)
    df_disambig_calcul = calcul_bias_target_n_persona(df_disambig)
    df_disambig_abs_calcul = calcul_bias_target_n_persona(df_disambig_abs)





    df_ambig_calcul = refine_column_names(df_ambig_calcul, 'polarity_amb')
    df_ambig_abs_calcul = refine_column_names(df_ambig_abs_calcul, 'amount_amb')
    df_disambig_calcul = refine_column_names(df_disambig_calcul, 'polarity_dis')
    df_disambig_abs_calcul = refine_column_names(df_disambig_abs_calcul, 'amount_dis')

    df_merged = pd.concat([df_overall_calcul, df_ambig_calcul,df_disambig_calcul,  df_ambig_abs_calcul, df_disambig_abs_calcul], axis=1)

    dir_save = os.path.join(args.save_dir, args.result_dir, args.model, args.category)
    dir_checker(dir_save)
    file_name = 'merged_total_rp_{}_cc_{}.csv'.format(rp, cc)
    save_path = os.path.join(dir_save, file_name)

    df_merged.to_csv(save_path)
    print("FILE SAVED: {}".format(save_path))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, default='Bias_Score')
    #parser.add_argument('--result_dir', type=str, default='Bias_Score_notunknown')
    parser.add_argument('--save_dir', type=str, default='total_merged')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--model', type=str, default='meta-llama/Llama-2-70b-chat-hf')

    parser.add_argument('--category', type=str, default='Sexual_orientation')

    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    #print(args)

    points = [(2,1), (1,1), (1,0)]
    cats = ['Age', 'Religion', 'Sexual_orientation',  'SES','Race_ethnicity', ] # ]#

    for cat in cats:
        args.category = cat
        for point in points:
            args.rp = point[0]
            args.cc = point[1]
            print(args)
            main(args)
