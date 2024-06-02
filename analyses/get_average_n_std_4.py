import os, argparse
import pandas as pd
import numpy as np

def average(args, domain, instruction_k):
    source_dir = os.path.join(args.source_dir, args.model, domain)

    for cont in args.context:
        df_list = tuple()
        for inst_no in range(instruction_k):
            f_name = args.source_file.format(inst_no, domain, cont, args.rp, args.cc)
            f_path = os.path.join(source_dir, f_name)

            df = pd.read_csv(f_path, index_col=0)
            df_list += (df.values,)

        index = df.index.values.tolist()
        column = df.columns.values.tolist()

        mean = np.dstack(df_list).mean(axis=2)
        std = np.dstack(df_list).std(axis=2)

        df_mean = pd.DataFrame(mean, index=index, columns=column)
        df_std = pd.DataFrame(std, index=index, columns=[col+'_std' for col in column])

        df_merged = pd.concat([df_mean, df_std], axis=1)

        # save df_merged
        save_dir = os.path.join(args.source_dir, args.model, domain)
        save_file = args.save_file.format(domain, cont, args.rp, args.cc)
        save_path = os.path.join(save_dir, save_file)
        df_merged.to_csv(save_path, index=True)
        print("FILE SAVED: {}".format(save_path))


def main(args):
    domains = ['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation']
    # toy
    #domains = ['Age']
    for d in domains:
        instruction_k = args.instruction_k
        if ('Llama' in args.model) & (d in ['SES', 'Race_ethnicity']):
            instruction_k = 3


        average(args, d, instruction_k)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./total_score')
    parser.add_argument('--source_file', type=str, default='inst_{}_{}_{}_rp_{}_cc_{}.csv')   # inst_{inst_no}_{domain}_context}_rp_{}_cc_{}.csv
    parser.add_argument('--save_file', type=str, default='aver_{}_{}_rp_{}_cc_{}.csv')

    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--instruction_k', type=int, default=5)
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--instruction_k', type=int, default=1)
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--instruction_k', type=int, default=5)

    parser.add_argument('--context', type=list, default=['ambig', 'disambig'])

    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()




if __name__ == "__main__":
    args = get_args()
    main(args)