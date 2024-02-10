import os, argparse
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

def main(args):

    tb_amb = pd.DataFrame()
    tb_dis = pd.DataFrame()
    bamt_amb = pd.DataFrame()
    bamt_dis = pd.DataFrame()
    pb_amb = pd.DataFrame()
    pb_dis = pd.DataFrame()

    for category in args.categories:
        f_name = args.file_name.format(args.rp, args.cc)
        f_path = os.path.join(args.result_dir, category, f_name)

        df = pd.read_csv(f_path)

        Polarity


def concat(df_total_amb, df_total_dis, df, score_name, contexts):
    amb = df.loc[:, ['{}_{}'.format(score_name, contexts[0])]].T
    dis = df[['{}_{}'.format(score_name, contexts[1])]].T
    #print(amb.T)

    df_total_amb = pd.concat([df_total_amb, amb], axis=0, ignore_index=True)
    df_total_dis = pd.concat([df_total_dis, dis], axis=0, ignore_index=True)

    return df_total_amb, df_total_dis


def collect_tables(args):
    df_tb_amb = pd.DataFrame()
    df_tb_dis = pd.DataFrame()
    df_bamt_amb = pd.DataFrame()
    df_bamt_dis = pd.DataFrame()
    df_pb_amb = pd.DataFrame()
    df_pb_dis = pd.DataFrame()

    for category in args.categories:
        f_name = args.file_name.format(args.rp, args.cc)
        f_path = os.path.join(args.result_dir, category, f_name)

        df = pd.read_csv(f_path)

        #scores = ['Polarity', 'Amount', 'BS']
        contexts = ['ambig', 'disambig']

        df_tb_amb, df_tb_dis = concat(df_tb_amb, df_tb_dis, df, 'Polarity', contexts)
        df_bamt_amb, df_bamt_dis = concat(df_bamt_amb, df_bamt_dis, df, 'Amount', contexts)
        df_pb_amb, df_pb_dis = concat(df_pb_amb, df_pb_dis, df, 'PB', contexts)

    def rename_index_col(df, rows, cols):
        df.index=rows
        df.columns=cols
        return df

    df_tb_amb = rename_index_col(df_tb_amb, args.cat_names, args.model_names)
    df_tb_dis = rename_index_col(df_tb_dis, args.cat_names, args.model_names)
    df_bamt_amb = rename_index_col(df_bamt_amb, args.cat_names, args.model_names)
    df_bamt_dis = rename_index_col(df_bamt_dis, args.cat_names, args.model_names)
    df_pb_amb = rename_index_col(df_pb_amb, args.cat_names, args.model_names)
    df_pb_dis = rename_index_col(df_pb_dis, args.cat_names, args.model_names)

    df_tb_amb.to_csv(os.path.join(args.save_dir, 'target_bias_ambig.csv'))
    df_tb_dis.to_csv(os.path.join(args.save_dir, 'target_bias_disambig.csv'))
    df_bamt_amb.to_csv(os.path.join(args.save_dir, 'bias_amount_ambig.csv'))
    df_bamt_dis.to_csv(os.path.join(args.save_dir, 'bias_amount_disambig.csv'))
    df_pb_amb.to_csv(os.path.join(args.save_dir, 'persona_bias_ambig.csv'))
    df_pb_dis.to_csv(os.path.join(args.save_dir, 'persona_bias_disambig.csv'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, default='./../total_merged_models/Bias_Score_modifyunknown')
    parser.add_argument('--file_name', type=str, default='merged_total_models_rp_{}_cc_{}.csv')
    parser.add_argument('--save_dir', type=str, default='./result_tables')
    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    parser.add_argument('--categories', type=list, default=['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation'])
    parser.add_argument('--cat_names', type=list, default=['Age', 'Race', 'Religion', 'SES', 'Sexual orientation'])
    parser.add_argument('--model_names', type=list, default=['llama-2-7b', 'llama-2-13b', 'llama-2-70b', 'gpt-3.5-turbo', 'gpt-4-turbo'])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    collect_tables(args)
    #main(args)