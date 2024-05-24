import os, glob, argparse
import json
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False
import seaborn as sns


def set_tex():
    os.environ['PATH'] += os.pathsep + "/usr/bin/latex"
    print(os.getenv("PATH"))

    plt.rcParams.update({
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    })


def get_persona_list(category):
    sorter = []

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
        sorter = ['Atheist', 'Christian', 'Protestant', 'Catholic', 'Mormon', 'Jewish', 'Muslim', 'Sikh', 'Hindu',
                  'Buddhist']
    elif category == 'SES':
        sorter = ['chemical engineer', 'chemist', 'dentist', 'judge', 'lawyer', 'optometrist', 'pharmacist',
                  'physician', 'physics teacher', 'professor', 'psychologist', 'scientist',
                  'bartender', 'cashier', 'construction worker', 'farm worker', 'janitor', 'line cook',
                  'parking attendant', 'receptionist', 'sales clerk', 'server', 'taxi driver', 'truck driver', ]
    elif category == 'Sexual_orientation':
        sorter = ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual']

    return sorter


def get_target_list(category):
    targets = []
    if category == "Age":
        targets = ['nonOld', 'old']
    if category == "Race_ethnicity":
        targets = ['African', 'African American', 'Black', 'Caucasian', 'White', 'European', 'Roma', 'Native American',
                   'South American', 'Hispanic', 'Latino', 'Jewish', 'Arab', 'Middle Eastern', 'Asian']
    if category == "Religion":
        targets = ["Atheist", "Christian", "Protestant", "Catholic", "Mormon", "Jewish", "Muslim", "Sikh", "Hindu",
                   "Buddhist"]
    if category == "SES":
        targets = ['highSES', 'lowSES']
    if category == "Sexual_orientation":
        targets = ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual']
    return targets


def get_min_max(category, context, metric):
        if category == "Age":
            if context == "ambig":
                if metric == "tbti":
                    return -11, 19
                if metric == "tb":
                    return 0, 15
                if metric == 'bamt':
                    return 0, 140
                if metric == 'pb':
                    return 0, 27
            else:   # disambig
                if metric == "tbti":
                    return -1, 8
                if metric == "tb":
                    return 0, 6
                if metric == 'bamt':
                    return 0, 62
                if metric == 'pb':
                    return 0, 5
        if category == "Race_ethnicity":
            if context == "ambig":
                if metric == "tbti":
                    return -14, 46
                if metric == "tb":
                    return 0, 12
                if metric == 'bamt':
                    return 0, 100
                if metric == 'pb':
                    return 0, 10
            else:
                if metric == "tbti":
                    return -12, 15
                if metric == "tb":
                    return 0, 6
                if metric == 'bamt':
                    return 0, 47
                if metric == 'pb':
                    return 0, 7
        if category == "Religion":
            if context == "ambig":
                if metric == "tbti":
                    return -38, 66
                if metric == "tb":
                    return 0, 19
                if metric == 'bamt':
                    return 0, 101
                if metric == 'pb':
                    return 0, 17
            else:
                if metric == "tbti":
                    return -15, 21
                if metric == "tb":
                    return 0, 7
                if metric == 'bamt':
                    return 0, 43
                if metric == 'pb':
                    return 0, 11
        if category == "SES":
            if context == "ambig":
                if metric == "tbti":
                    return -31, 38
                if metric == "tb":
                    return 0, 34
                if metric == 'bamt':
                    return 0, 130
                if metric == 'pb':
                    return 0, 13
            else:
                if metric == "tbti":
                    return -4, 5
                if metric == "tb":
                    return 0, 4
                if metric == 'bamt':
                    return 0, 53
                if metric == 'pb':
                    return 0, 4
        if category == "Sexual_orientation":
            if context == "ambig":
                if metric == "tbti":
                    return -10, 34
                if metric == "tb":
                    return 0, 13
                if metric == 'bamt':
                    return 0, 89
                if metric == 'pb':
                    return 0, 16
            else:
                if metric == "tbti":
                    return -8, 15
                if metric == "tb":
                    return 0, 5
                if metric == 'bamt':
                    return 0, 47
                if metric == 'pb':
                    return 0, 6


def result2(args):
    dir = args.source_dir2
    model = args.model

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    for cat in args.categories:
        personas = get_persona_list(cat)
        targets = get_target_list(cat)

        for context in ['ambig', 'disambig']:

            f_name = args.file_name_2.format(cat, context, args.rp, args.cc)

            file_path = os.path.join(dir, model, cat, f_name)

            df = pd.read_csv(file_path, index_col=0)
            #print(df)

            tb_ti = df[['TB_{}'.format(t) for t in targets]]
            #bamt_ti = df[['BAmt_{}'.format(t) for t in targets]]
            tb = df[['TB']]
            pb = df[['PB_p']]
            bamt = df[['BAmt']]
            #bs = df[['BS_{}'.format('a' if context == 'ambig' else 'd')]]

            df_figure = pd.concat([tb_ti, tb, bamt, pb], axis=1 ) *100  # tb,  bamt_ti, bamt
            df_figure = df_figure.reindex(['Baseline'] + personas)
            df_figure.index = ['Default'] + personas
            print(df_figure)

            # tex/latex font


            n_target = len(targets)
            n_persona = len(df_figure)
            if cat == 'Age' or cat == 'Sexual_orientation':
                fig, ax = plt.subplots(figsize=(((n_target) + 3) / 2 + 1, (n_persona + 1)/3+1))
            elif cat == 'SES':
                fig, ax = plt.subplots(figsize=(((n_target) + 3) / 2 + 1.5, (n_persona + 1) / 3))
            else:
                fig, ax = plt.subplots(figsize=(((n_target) + 3) / 2 + 1, (n_persona + 1) / 3))


            xticks = [] + targets; xticks.extend([r"$\mathrm{TB}_{p→T}$", r"$\mathrm{BAmt}_{p→T}$", r"$\mathrm{PB}_{p}$"])

            # tb_ti
            start = 0; end = n_target
            data_tb_ti = df_figure.copy()
            data_tb_ti.iloc[:, end:] = float('nan')

            vmin, vmax = get_min_max(cat, context, 'tbti')
            print(vmin, vmax)
            absmax = max(abs(vmin), vmax)
            vmin = -absmax; vmax = absmax
            print(vmin, vmax)

            ax = sns.heatmap(data_tb_ti, annot=data_tb_ti.round(0), annot_kws={"fontsize": 'large'},
                             cmap=args.cmap_tb_i, cbar=False,
                             vmin=vmin, vmax=vmax,
                             xticklabels=xticks,
                             )
            # vmin=vmin, vmax=vmax)

            # TB
            start = end; end = end +1
            data_tb = df_figure.copy()
            data_tb.iloc[:, :start] = float('nan')
            data_tb.iloc[:, end:] = float('nan')
            vmin, vmax = get_min_max(cat, context, 'tb')
            sns.heatmap(data_tb, annot=data_tb.round(0),  annot_kws={"fontsize": 'large'},
                        cmap=args.cmap_tb, cbar=False,
                        vmin=vmin, vmax=vmax)

            # BAmt
            start = end; end = end + 1
            data_bamt = df_figure.copy()
            data_bamt.iloc[:, :start] = float('nan')
            data_bamt.iloc[:, end:] = float('nan')
            vmin, vmax = get_min_max(cat, context, 'bamt')
            sns.heatmap(data_bamt, annot=data_bamt.round(0), annot_kws={"fontsize": 'large'}, fmt=".0f",
                        cmap=args.cmap_bamt, cbar=False,
                        vmin=vmin, vmax=vmax)

            # PB
            start = end; end = end +1
            data_pb = df_figure.copy()
            data_pb.iloc[:, :start] = float('nan')
            data_pb.iloc[:, end:] = float('nan')
            vmin, vmax = get_min_max(cat, context, 'pb')
            sns.heatmap(data_pb, annot=data_pb.round(0),  annot_kws={"fontsize": 'large'},
                        cmap=args.cmap_pb, cbar=False,
                        vmin=vmin, vmax=vmax,
                        xticklabels=xticks)
            '''
            start = end; end = end+n_target
            data_bamt_ti = df_figure.copy()
            data_bamt_ti.iloc[:, :start] = float('nan')
            data_bamt_ti.iloc[:, end:] = float('nan')
            sns.heatmap(data_bamt_ti, annot=data_bamt_ti.round(3), cmap = args.cmap_bamt, cbar=False)


            start = end; end = end+1
            data_bamt = df_figure.copy()
            data_bamt.iloc[:, :start] = float('nan')
            data_bamt.iloc[:, end:] = float('nan')
            sns.heatmap(data_bamt, annot=data_bamt.round(3), cmap = args.cmap_bamt, cbar=False)


            start = end; end = end + 1
            data_bs = df_figure.copy()
            data_bs.iloc[:, :start] = float('nan')
            data_bs.iloc[:, end:] = float('nan')
            sns.heatmap(data_bs, annot=data_bs.round(0),
                        cmap=args.cmap_bs, cbar=False, center=0,
                        xticklabels=xticks)
            '''

            # tex font
            set_tex()

            plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = False, labeltop=True, rotation=10)
            plt.xticks(rotation=90, fontsize='large')
            plt.yticks(rotation=0, fontsize='large')
            plt.xlabel("Target", fontsize='large')
            ax.xaxis.set_label_position('top')
            plt.ylabel("Persona", fontsize='large' )

            ax.hlines([1], *ax.get_xlim(), colors='white', linewidth=2)
            ymin, ymax = plt.ylim()
            ax.vlines(x=n_target, ymin=ymin, ymax=ymax, colors='white', linewidth=2)

            plt.tight_layout()

            save_dir = './result_2_TB_ti/{}'.format(cat)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                os.makedirs(os.path.join(save_dir, 'meta-llama'))
            save_file = '{}_{}_{}.pdf'.format(model, cat, context)
            plt.savefig(os.path.join(save_dir, save_file), dpi=200)

            plt.show()

            # initiallize tex setting
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def result2_for_case(args):
    dir = args.source_dir2
    model = 'gpt-3.5-turbo-0613'

    for cat in ['Religion']:
        personas = get_persona_list(cat)
        targets = get_target_list(cat)

        for context in ['ambig', 'disambig']:

            f_name = args.file_name_2.format(cat, context, args.rp, args.cc)

            file_path = os.path.join(dir, model, cat, f_name)

            df = pd.read_csv(file_path, index_col=0)
            # print(df)

            tb_ti = df[['TB_{}'.format(t) for t in targets]]
            # bamt_ti = df[['BAmt_{}'.format(t) for t in targets]]
            tb = df[['TB']]
            pb = df[['PB_p']]
            bamt = df[['BAmt']]
            # bs = df[['BS_{}'.format('a' if context == 'ambig' else 'd')]]

            df_figure = pd.concat([tb_ti,], axis=1) # tb,  bamt_ti, bamt
            df_figure = df_figure.reindex(['Baseline'] + personas)
            print(df_figure)

            n_target = len(targets)
            n_persona = len(df_figure)
            if cat != 'SES':
                fig, ax = plt.subplots(figsize=((n_target) / 2 + 1, (n_persona +1) / 3))
            else:
                fig, ax = plt.subplots(figsize=((n_target) / 2 + 1, (n_persona +1) / 3))
            xticks = [] + targets
            yticks = ['Default'] + personas
            # xticks.extend(["TB_s", "BAmt_s", "PB_s"])

            # tb_ti
            start = 0;
            end = n_target
            data_tb_ti = df_figure.copy()
            data_tb_ti.iloc[:, end:] = float('nan')

            vmin, vmax = get_min_max(cat, context, 'tbti')
            vmin /= 100
            vmax /= 100
            print(vmin, vmax)
            absmax = max(abs(vmin), vmax)
            vmin = -absmax;
            vmax = absmax
            print(vmin, vmax)


            ax = sns.heatmap(data_tb_ti, annot=data_tb_ti.round(2), annot_kws={"fontsize": 'large'},
                             cmap=args.cmap_tb_i, cbar=False,
                             vmin=vmin, vmax=vmax,
                             xticklabels=xticks,
                             yticklabels=yticks,
                             )
            # vmin=vmin, vmax=vmax)

            '''
            # TB
            start = end;
            end = en
            d + 1
            data_tb = df_figure.copy()
            data_tb.iloc[:, :start] = float('nan')
            data_tb.iloc[:, end:] = float('nan')
            vmin, vmax = get_min_max(cat, context, 'tb')
            sns.heatmap(data_tb, annot=data_tb.round(0), annot_kws={"fontsize": 'large'},
                        cmap=args.cmap_tb, cbar=False,
                        vmin=vmin, vmax=vmax)

            # BAmt
            start = end;
            end = end + 1
            data_bamt = df_figure.copy()
            data_bamt.iloc[:, :start] = float('nan')
            data_bamt.iloc[:, end:] = float('nan')
            vmin, vmax = get_min_max(cat, context, 'bamt')
            sns.heatmap(data_bamt, annot=data_bamt.round(0), annot_kws={"fontsize": 'large'}, fmt=".0f",
                        cmap=args.cmap_bamt, cbar=False,
                        vmin=vmin, vmax=vmax)

            # PB
            start = end;
            end = en
            d + 1
            data_pb = df_figure.copy()
            data_pb.iloc[:, :start] = float('nan')
            data_pb.iloc[:, end:] = float('nan')
            vmin, vmax = get_min_max(cat, context, 'pb')
            sns.heatmap(data_pb, annot=data_pb.round(0), annot_kws={"fontsize": 'large'},
                        cmap=args.cmap_pb, cbar=False,
                        vmin=vmin, vmax=vmax,
                        xticklabels=xticks)
            
            start = end; end = end+n_target
            data_bamt_ti = df_figure.copy()
            data_bamt_ti.iloc[:, :start] = float('nan')
            data_bamt_ti.iloc[:, end:] = float('nan')
            sns.heatmap(data_bamt_ti, annot=data_bamt_ti.round(3), cmap = args.cmap_bamt, cbar=False)


            start = end; end = end+1
            data_bamt = df_figure.copy()
            data_bamt.iloc[:, :start] = float('nan')
            data_bamt.iloc[:, end:] = float('nan')
            sns.heatmap(data_bamt, annot=data_bamt.round(3), cmap = args.cmap_bamt, cbar=False)


            start = end; end = end + 1
            data_bs = df_figure.copy()
            data_bs.iloc[:, :start] = float('nan')
            data_bs.iloc[:, end:] = float('nan')
            sns.heatmap(data_bs, annot=data_bs.round(0),
                        cmap=args.cmap_bs, cbar=False, center=0,
                        xticklabels=xticks)
            '''

            # tex font
            set_tex()

            plt.tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False, labeltop=True,
                            rotation=10)
            plt.xticks(rotation=50, fontsize='large')
            plt.yticks(rotation=0, fontsize='large')
            plt.xlabel("Target", fontsize='large')
            ax.xaxis.set_label_position('top')
            plt.ylabel("Persona", fontsize='large')

            ax.hlines([1], *ax.get_xlim(), colors='white', linewidth=2)
            ymin, ymax = plt.ylim()
            ax.vlines(x=n_target, ymin=ymin, ymax=ymax, colors='white', linewidth=2)

            plt.tight_layout()

            save_dir = './result_2_TB_ti/tbti_case'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = '{}_{}_{}_only_TBti.pdf'.format(model, cat, context)
            plt.savefig(os.path.join(save_dir, save_file), dpi=200)

            plt.show()

            # initiallize tex setting
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)



if __name__ == "__main__":
    # print(os.environ['PATH'])
    print("plot_diagonal_heatmap.py")