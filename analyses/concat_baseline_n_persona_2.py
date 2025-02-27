import os, argparse
import glob
import pandas as pd


def save_concat_df(args, df, instruction_k, domain, criterion, context, rp, cc):
    # inst_{instruction_k}_{domain}_{context}_{score criteria}_rp_{rp}_cc_{cc}.csv
    if criterion == 'score':
        cri = 'tb'
    elif criterion == 'abs_score':
        cri = 'bamt'
    else:
        raise(ValueError("No valid criterion: {}".format(criterion)))

    save_dir = os.path.join(args.result_dir, args.model, domain)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_name = args.result_file.format(instruction_k,
                                             domain,
                                             context,
                                             cri,
                                             rp, cc)
    save_path = os.path.join(save_dir, save_file_name)

    df.to_csv(save_path, index=True)
    print("FILE SAVED: {}".format(save_path))


def concat_files(args, domain, instruction_k):
    rp, cc = args.rp, args.cc
    # path
    dir_source = os.path.join(args.source_dir, args.model)
    dir_source_base = os.path.join(dir_source, 'Baseline')
    dir_source_domain = os.path.join(dir_source, domain)

    for k in range(instruction_k):
        for criterion in args.criteria:
            for context in args.context:
                f_name = 'inst_{}_*2{}_{}_{}_rp_{}_cc_{}.csv'.format(k, domain, context, criterion, rp, cc)


                f_name_base = glob.glob(os.path.join(dir_source_base, f_name))[0]
                f_name_domain = glob.glob(os.path.join(dir_source_domain, f_name))[0]

                #print(f_name_base)
                #print(f_name_domain)
                #print()

                df_base = pd.read_csv(f_name_base, index_col=0)
                df_domain = pd.read_csv(f_name_domain, index_col=0)

                def remove_item(df, items):
                    index = df.index.values.tolist()
                    columns = df.columns.values.tolist()

                    for item in items:
                        if item in index: index.remove(item)
                        if item in columns: columns.remove(item)

                    #print(index)
                    #print(columns)
                    # slicing
                    df = df.loc[index, :]
                    df = df[columns]
                    #print(df.shape)
                    #print(df)

                    return df


                if domain == 'Race_ethnicity':
                    df_base = remove_item(df_base, ['Alaskan'])
                    df_domain = remove_item(df_domain, ['Alaskan'])
                elif domain == 'Religion':
                    df_base = remove_item(df_base, ['Orthodox'])
                    df_domain = remove_item(df_domain, ['Orthodox'])

                df_concat = pd.concat([df_base, df_domain], axis=0)
                #print(df_concat)

                save_concat_df(args, df_concat, k, domain, criterion, context, rp, cc)


def main(args):
    points = [(2, 1), (1, 1), (1, 0)]
    domains = ['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation']
    # toy
    #domains = ['Race_ethnicity', 'Religion']
    for p in points:
        args.rp = p[0]
        args.cc = p[1]
        for d in domains:
            instruction_k = args.instruction_k
            if ('Llama' in args.model) & (d in ['SES', 'Race_ethnicity']):
                instruction_k = 3

            concat_files(args, d, instruction_k)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./Bias_Score')
    parser.add_argument('--result_dir', type=str, default='./Bias_Score_concatenated')
    parser.add_argument('--result_file', type=str, default='inst_{}_{}_{}_{}_rp_{}_cc_{}.csv')  # inst_{instruction_k}_{domain}_{context}_{score criteria}_rp_{rp}_cc_{cc}.csv


    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--instruction_k', type=int, default=5)
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--instruction_k', type=int, default=1)
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--instruction_k', type=int, default=5)


    parser.add_argument('--criteria', type=list, default=['score', 'abs_score'])    # TB, BAmt
    parser.add_argument('--context', type=list, default=['ambig', 'disambig'])
    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)