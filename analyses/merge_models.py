import os, argparse
import pandas as pd


def get_file_path(args, model, category, rp, cc):
    dir = os.path.join(args.result_dir, model, category)
    f_name = 'merged_total_rp_{}_cc_{}.csv'.format(rp, cc)
    return os.path.join(dir, f_name)


def save_df(args, df):
    def dir_checker(path):
        if not os.path.exists(path):
            os.makedirs(path)
    dir = os.path.join(args.save_dir, args.category)
    dir_checker(dir)
    f_name = 'merged_total_models_rp_{}_cc_{}.csv'.format(args.rp, args.cc)
    save_path = os.path.join(dir, f_name)


    df.to_csv(save_path, index=False)
    print("FILE SAVED: {}".format(save_path))


def aver_TB(df):
    TB_all_ambig = df.loc[:, 'TB_ambig'].sum()
    TB_all_disambig = df.loc[:, 'TB_disambig'].sum()
    n = len(df)
    return TB_all_ambig/n, TB_all_disambig/n

def aver_PB(df):
    PB_all_ambig = df.loc[:, 'PB_ambig'].sum()
    PB_all_disambig = df.loc[:, 'PB_disambig'].sum()
    n = len(df)-1
    return PB_all_ambig/n, PB_all_disambig/n

def main(args, models):
    category = args.category
    rp, cc = args.rp, args.cc

    total_df = pd.DataFrame()
    for model in models:
        file_path = get_file_path(args, model, category, rp, cc)
        df = pd.read_csv(file_path, index_col=0)
        #print(df)

        baseline = df.loc['Baseline', :]
        pol_base_ambig = baseline['TB_polarity_amb']
        amt_base_ambig = baseline['TB_amount_amb']
        pol_base_disambig = baseline['TB_polarity_dis']
        amt_base_disambig = baseline['TB_amount_dis']

        #pol_base_disambig = df.at['Baseline', 'TB_polarity_dis']

        #TB_all_ambig, TB_all_disambig = aver_TB(df)
        PB_all_ambig, PB_all_disambig = aver_PB(df)

        item = {
            "Model": model,
            "TB_0_ambig": TB_base_ambig, "TB_all_ambig": TB_all_ambig, "PB_all_ambig": PB_all_ambig,
            "TB_0_disambig": TB_base_disambig, "TB_all_disambig": TB_all_disambig, "PB_all_disambig": PB_all_disambig,
        }
        df_item = pd.DataFrame().from_dict([item])
        total_df = pd.concat([total_df, df_item], ignore_index=True, axis=0)
    save_df(args, total_df)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, default='total_merged/Bias_Score')
    parser.add_argument('--save_dir', type=str, default='total_merged_models/Bias_Score')
    #parser.add_argument('--result_dir', type=str, default='total_merged/Bias_Score_notunknown')
    #parser.add_argument('--save_dir', type=str, default='total_merged_models/Bias_Score_notunknown')

    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--model', type=str, default='meta-llama/Llama-2-70b-chat-hf')

    parser.add_argument('--category', type=str, default='Sexual_orientation')

    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    #print(args)

    models = ['gpt-3.5-turbo-0613', 'gpt-4-1106-preview',
              'meta-llama/Llama-2-70b-chat-hf']# 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf']
    points = [(2,1), (1,1), (1,0)]
    cats = ['Age', 'Religion', 'Sexual_orientation', 'Race_ethnicity', 'SES']

    for cat in cats:
        args.category = cat
        for point in points:
            args.rp = point[0]
            args.cc = point[1]
            print(args)
            main(args, models)
