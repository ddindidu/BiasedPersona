import os, glob, argparse
import json
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def result2(args):
    def get_target_list(category):
        targets = []
        if category == "Age":
            targets = ['nonOld', 'old']
        if category == "Race_ethnicity":
            targets = ['African', 'African American', 'Black', 'Caucasian', 'White', 'European', 'Roma', 'Native American', 'South American', 'Hispanic', 'Latino', 'Jewish', 'Arab', 'Middle Eastern', 'Asian']
        if category == "Religion":
            targets = ["Atheist", "Christian", "Protestant", "Catholic",  "Mormon", "Jewish", "Muslim", "Sikh", "Hindu", "Buddhist"]
        if category == "SES":
            targets = ['highSES', 'lowSES']
        if category == "Sexual_orientation":
            targets = ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual']
        return targets

    def get_min_max(category, context, metric):
        if category == "Age":
            if context == "amb":
                if metric == "tbti":
                    return -11, 19
                if metric == "tb":
                    return 0, 16
                if metric == 'bamt':
                    return 0, 140
                if metric == 'pb':
                    return 0, 26
            else:
                if metric == "tbti":
                    return -1, 7
                if metric == "tb":
                    return 0, 7
                if metric == 'bamt':
                    return 0, 62
                if metric == 'pb':
                    return 0, 5
        if category == "Race_ethnicity":
            if context == "amb":
                if metric == "tbti":
                    return -27, 46
                if metric == "tb":
                    return 0, 11
                if metric == 'bamt':
                    return 0, 100
                if metric == 'pb':
                    return 0, 10
            else:
                if metric == "tbti":
                    return -18, 29
                if metric == "tb":
                    return 0, 6
                if metric == 'bamt':
                    return 0, 48
                if metric == 'pb':
                    return 0, 8
        if category == "Religion":
            if context == "amb":
                if metric == "tbti":
                    return -38, 66
                if metric == "tb":
                    return 0, 20
                if metric == 'bamt':
                    return 0, 105
                if metric == 'pb':
                    return 0, 15
            else:
                if metric == "tbti":
                    return -23, 21
                if metric == "tb":
                    return 0, 8
                if metric == 'bamt':
                    return 0, 51
                if metric == 'pb':
                    return 0, 13
        if category == "SES":
            if context == "amb":
                if metric == "tbti":
                    return -32, 38
                if metric == "tb":
                    return 0, 35
                if metric == 'bamt':
                    return 0, 132
                if metric == 'pb':
                    return 0, 13
            else:
                if metric == "tbti":
                    return -4, 6
                if metric == "tb":
                    return 0, 5
                if metric == 'bamt':
                    return 0, 53
                if metric == 'pb':
                    return 0, 5
        if category == "Sexual_orientation":
            if context == "amb":
                if metric == "tbti":
                    return -19, 34
                if metric == "tb":
                    return 0, 13
                if metric == 'bamt':
                    return 0, 89
                if metric == 'pb':
                    return 0, 16
            else:
                if metric == "tbti":
                    return -15, 15
                if metric == "tb":
                    return 0, 8
                if metric == 'bamt':
                    return 0, 47
                if metric == 'pb':
                    return 0, 9

    dir = args.result2_dir
    model = args.model

    for cat in args.categories:

        f_name = args.file_name_2.format(args.rp, args.cc)

        file_path = os.path.join(dir, model, cat, f_name)

        df = pd.read_csv(file_path, index_col=0)

        targets = get_target_list(cat)

        for context in ['amb', 'dis']:
            scores = ['polarity', 'amount']

            tb_ti = df[['{}_{}_{}'.format(t, scores[0], context) for t in targets]]
            bamt_ti = df[['{}_{}_{}'.format(t, scores[1], context) for t in targets]]
            tb = df[['TB_{}_{}'.format(scores[0], context)]]
            pb = df[['PB_{}_{}'.format(scores[0], context)]]
            bamt = df[['TB_{}_{}'.format(scores[1], context)]]
            bs = df[['BS_{}'.format('a')]]

            df_figure = pd.concat([tb_ti, tb, bamt, pb], axis=1 ) *100  # tb,  bamt_ti, bamt
            print(df_figure)


            n_target = len(targets)
            n_persona = len(df_figure)
            if cat == 'Age' or cat == 'Sexual_orientation':
                fig, ax = plt.subplots(figsize=(((n_target) + 3) / 2 + 1, (n_persona + 1)/3+1))

            else:
                fig, ax = plt.subplots(figsize=(((n_target) + 3) / 2 + 1, (n_persona + 1) / 3))




            xticks = [] + targets; xticks.extend(["TB_{p→T}", "BAmt_{p→T}", "PB_p"])

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

            save_dir = './result_2_TB_ti'
            save_file = '{}_{}_{}.png'.format(model, cat, context)
            plt.savefig(os.path.join(save_dir, save_file), dpi=200)

            plt.show()


def result2_for_case(args):
    def get_target_list(category):
        targets = []
        if category == "Age":
            targets = ['nonOld', 'old']
        if category == "Race_ethnicity":
            targets = ['African', 'African American', 'Black', 'Caucasian', 'White', 'European', 'Roma',
                       'Native American', 'South American', 'Hispanic', 'Latino', 'Jewish', 'Arab', 'Middle Eastern',
                       'Asian']
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
            if context == "amb":
                if metric == "tbti":
                    return -11, 19
                if metric == "tb":
                    return 0, 16
                if metric == 'bamt':
                    return 0, 140
                if metric == 'pb':
                    return 0, 26
            else:
                if metric == "tbti":
                    return -1, 7
                if metric == "tb":
                    return 0, 7
                if metric == 'bamt':
                    return 0, 62
                if metric == 'pb':
                    return 0, 5
        if category == "Race_ethnicity":
            if context == "amb":
                if metric == "tbti":
                    return -27, 46
                if metric == "tb":
                    return 0, 11
                if metric == 'bamt':
                    return 0, 100
                if metric == 'pb':
                    return 0, 10
            else:
                if metric == "tbti":
                    return -18, 29
                if metric == "tb":
                    return 0, 6
                if metric == 'bamt':
                    return 0, 48
                if metric == 'pb':
                    return 0, 8
        if category == "Religion":
            if context == "amb":
                if metric == "tbti":
                    return -38, 66
                if metric == "tb":
                    return 0, 20
                if metric == 'bamt':
                    return 0, 105
                if metric == 'pb':
                    return 0, 15
            else:
                if metric == "tbti":
                    return -23, 21
                if metric == "tb":
                    return 0, 8
                if metric == 'bamt':
                    return 0, 51
                if metric == 'pb':
                    return 0, 13
        if category == "SES":
            if context == "amb":
                if metric == "tbti":
                    return -32, 38
                if metric == "tb":
                    return 0, 35
                if metric == 'bamt':
                    return 0, 132
                if metric == 'pb':
                    return 0, 13
            else:
                if metric == "tbti":
                    return -4, 6
                if metric == "tb":
                    return 0, 5
                if metric == 'bamt':
                    return 0, 53
                if metric == 'pb':
                    return 0, 5
        if category == "Sexual_orientation":
            if context == "amb":
                if metric == "tbti":
                    return -19, 34
                if metric == "tb":
                    return 0, 13
                if metric == 'bamt':
                    return 0, 89
                if metric == 'pb':
                    return 0, 16
            else:
                if metric == "tbti":
                    return -15, 15
                if metric == "tb":
                    return 0, 8
                if metric == 'bamt':
                    return 0, 47
                if metric == 'pb':
                    return 0, 9

    dir = args.result2_dir
    model = 'gpt-3.5-turbo-0613'

    for cat in ['Religion']:

        f_name = args.file_name_2.format(args.rp, args.cc)

        file_path = os.path.join(dir, model, cat, f_name)

        df = pd.read_csv(file_path, index_col=0)

        targets = get_target_list(cat)

        for context in ['amb', 'dis']:
            scores = ['polarity', 'amount']

            tb_ti = df[['{}_{}_{}'.format(t, scores[0], context) for t in targets]]
            bamt_ti = df[['{}_{}_{}'.format(t, scores[1], context) for t in targets]]
            tb = df[['TB_{}_{}'.format(scores[0], context)]]
            pb = df[['PB_{}_{}'.format(scores[0], context)]]
            bamt = df[['TB_{}_{}'.format(scores[1], context)]]
            bs = df[['BS_{}'.format('a')]]

            df_figure = pd.concat([tb_ti,], axis=1)  # tb,  bamt_ti, bamt
            print(df_figure)

            n_target = len(targets)
            n_persona = len(df_figure)
            if cat != 'SES':
                fig, ax = plt.subplots(figsize=((n_target) / 2 + 1, (n_persona +1) / 3))
            else:
                fig, ax = plt.subplots(figsize=((n_target) / 2 + 1, (n_persona +1) / 3))
            xticks = [] + targets;
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

            save_dir = './result_2_TB_ti'
            save_file = '{}_{}_{}_only_TBti.png'.format(model, cat, context)
            plt.savefig(os.path.join(save_dir, save_file), dpi=200)

            plt.show()



