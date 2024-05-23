import os, argparse
import pandas as pd


def merge(args, domain, context):
    df_total = pd.DataFrame()

    if context == 'ambig':
        cont = 'a'
    else:   # disambig
        cont = 'd'
    columns = ['TB', 'TB_std',
               'BAmt', 'BAmt_std',
               'PB', 'PB_std',
               'BS_{}'.format(cont),
               'BS_{}_std'.format(cont),
               'Acc_{}'.format(cont),
               'Acc_{}_std'.format(cont),
               ]

    for m in args.models:
        source_dir = os.path.join(args.source_dir, m, domain)
        source_file = args.source_file.format(domain, context, args.rp, args.cc)

        f_path = os.path.join(source_dir, source_file)

        df = pd.read_csv(f_path, index_col=0)
        df_selected = df.loc[['Baseline'], columns]
        #print(df_selected)
        df_selected.index = [m]

        df_total = pd.concat([df_total, df_selected], axis=0)

    return df_total




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./total_score')
    parser.add_argument('--source_file', type=str, default='aver_{}_{}_rp_{}_cc_{}.csv')
    parser.add_argument('--result_dir', type=str, default='./total_score_model_merged')
    parser.add_argument('--save_file', type=str, default='{}_{}_rp_{}_cc_{}.csv')   # domain_context_rp_cc

    parser.add_argument('--domains', type=str, default=['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation'])
    parser.add_argument('--contexts', type=list, default=['ambig', 'disambig'])
    parser.add_argument('--models', type=list,
                        default=['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf',
                                'gpt-3.5-turbo-0613', 'gpt-4-1106-preview',])

    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()



    for domain in args.domains:
        for context in args.contexts:
            result_df = merge(args, domain, context)

            save_dir = os.path.join(args.result_dir, domain)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = args.save_file.format(domain, context, args.rp, args.cc)
            save_path = os.path.join(save_dir, save_file)
            result_df.to_csv(save_path, index=True)
            print("FILE SAVED: {}".format(save_path))
