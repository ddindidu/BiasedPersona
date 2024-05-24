import os, argparse

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps

def result1(args, categories):
    Xs, Ys, pol, amt = [], [], [], []

    for y, zipped_info in enumerate(zip(categories, args.target_num)):
        cat, targ_n = zipped_info
        file_path = os.path.join(args.result_dir, cat, args.result_file)
        df = pd.read_csv(file_path)

        Xs.append(range(1, 6))
        Ys += 5 * [5-y]
        pol.extend(df['Polarity_ambig'].tolist())
        amt.extend((df['Amount_ambig']**2*5000).tolist())

    fig, ax = plt.subplots()
    #ax = fig.add_subplot()
    plt.scatter(Xs, Ys, c=pol, cmap='Reds', s=amt, edgecolors='black')


    plt.xlabel('Models')
    plt.show()


def result2(args, category):
    print(os.getcwd())
    file_path = os.path.join(args.persona_result_dir, category, 'merged_total_rp_2_cc_1.csv')
    assert os.path.exists(file_path), "NO FILE DIR"

    df = pd.read_csv(file_path, index_col=0)
    persona = df.index.tolist()
    persona = ['.', '^', '^', '^', '1', '1', 'x']
    TB = df['TB_polarity_amb'].tolist()
    TB_AMT = (df['TB_amount_amb']*50).tolist()
    PB = df['PB_polarity_amb'].tolist()
    BS = df['BS_a'].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # baseline
    ax.scatter(BS[:1], TB[:1], PB[:1], marker='o', s=TB_AMT[:1], label='baseline')
    ax.scatter(BS[1:4], TB[1:4], PB[1:4], marker='^',s=TB_AMT[1:4],  label='non old')
    ax.scatter(BS[4:6], TB[4:6], PB[4:6], marker='1', s=TB_AMT[4:6], label='middle-aged')
    ax.scatter(BS[6:], TB[6:], PB[6:], marker='x', s=TB_AMT[6:], label='old')

    ax.set_xlabel('BS')
    ax.set_ylabel('TB')
    ax.set_zlabel('PB')

    plt.show()





def main(args):
    file_path = os.path.join(args.result_dir, args.category, args.result_file)
    category = args.category

    #df = pd.read_csv(file_path)

    result1(args, args.categories)
    result2(args, args.category)


def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--persona_result_dir', type=str, default='./../total_score/gpt-3.5-turbo-0613')
    parser.add_argument('--result_dir', type=str, default='./../total_score_model_merged/')
    parser.add_argument('--result_file', type=str, default='{}_{}_rp_{}_cc_{}.csv')

    parser.add_argument('--models', type=list, default=['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'gpt-3.5-turbo-0613', 'gpt-4-1106-preview', ])
    parser.add_argument('--categories', type=list, default=['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation'])
    parser.add_argument('--category', type=str, default='Age')
    parser.add_argument('--target_num', type=list, default=[2, 15, 10, 2, 5])

    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)