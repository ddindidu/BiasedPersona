import os, argparse
import networkx as nx
from pyvis.network import Network

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps



def draw_network(args, category):
    print(os.getcwd())
    file_path = os.path.join(args.persona_result_dir, category, 'merged_total_rp_2_cc_1.csv')
    assert os.path.exists(file_path), "NO FILE DIR"

    df = pd.read_csv(file_path, index_col=0)
    G = nx.MultiDiGraph()

    def get_color(v):
        if v < 0:
            return 'red'
        elif v == 0:
            return 'black'
        else:
            return 'blue'

    G.add_edge('Baseline', 'Non Old', weight = df.at['Baseline', 'nonOld_polarity_amb'])
    G.add_edge('Baseline', 'Old', weight=df.at['Baseline', 'old_polarity_amb'])
    G.add_edge('Non Old', 'Non Old', weight=df.at['kid', 'nonOld_polarity_amb'])
    G.add_edge('Non Old', 'Old', weight=df.at['kid', 'old_polarity_amb'])
    G.add_edge('Old', 'Non Old', weight=df.at['elder', 'nonOld_polarity_amb'])
    G.add_edge('Old', 'Old', weight=df.at['elder', 'old_polarity_amb'])

    weights = nx.get_edge_attributes(G, 'weight').values()

    #colors = ['r' if w < 0 else 'b' for w in weights]
    weights = [abs(w)**2*100 for w in weights]

    '''
    nx.draw(G,
            edge_color = colors,
            width=list(weights),
            with_labels=True)
    '''
    nt = Network(directed=True)
    nt.from_nx(G)
    nt.set_edge_smooth('dynamic')

    for edge in nt.get_edges():
        print(edge)
        edge['color'] = get_color(edge['width'])
        edge['width'] = edge['width']**2*50

    nt.show('intro.html')
    plt.show()


def main(args):
    file_path = os.path.join(args.result_dir, args.category, args.result_file)
    category = args.category

    df = pd.read_csv(file_path)

    draw_network(args, category)


def get_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--persona_result_dir', type=str, default='./../total_merged/Bias_Score/gpt-3.5-turbo-0613')
    parser.add_argument('--result_dir', type=str, default='./../total_merged_models/Bias_Score')
    parser.add_argument('--result_file', type=str, default='merged_total_models_rp_2_cc_1.csv')

    parser.add_argument('--models', type=list, default=['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'gpt-3.5-turbo-0613', 'gpt-4-1106-preview', ])
    parser.add_argument('--categories', type=list, default=['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation'])
    parser.add_argument('--category', type=str, default='Age')
    parser.add_argument('--target_num', type=list, default=[2, 15, 10, 2, 5])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)