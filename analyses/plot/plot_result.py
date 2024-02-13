import os, glob, argparse
import json
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def result1_main_heatmap(args):
    file_tb = 'target_bias_{}_rp_{}_cc_{}.csv'
    file_bamt = 'bias_amount_{}_rp_{}_cc_{}.csv'
    file_pb = 'persona_bias_{}_rp_{}_cc_{}.csv'
    file_bs = 'bs_{}_rp_{}_cc_{}.csv'

    def get_df(dir, f_name, rp, cc):
        df_ambig = pd.read_csv(os.path.join(dir, f_name.format('ambig', rp, cc)), index_col=0)
        df_ambig.index=args.cat_names
        df_ambig.columns=args.model_names
        df_disambig = pd.read_csv(os.path.join(dir, f_name.format('disambig', rp, cc)), index_col=0)
        df_disambig.index=args.cat_names
        df_disambig.columns=args.model_names
        return df_ambig.T, df_disambig.T

    tb_amb, tb_dis = get_df(args.save_dir, file_tb, args.rp, args.cc)
    bamt_amb, bamt_dis = get_df(args.save_dir, file_bamt, args.rp, args.cc)
    pb_amb, pb_dis = get_df(args.save_dir, file_pb, args.rp, args.cc)
    bs_amb, bs_dis = get_df(args.save_dir, file_bs, args.rp, args.cc)

    def draw_heatmap(df, context, location, colormap, plusminus=False):
        if plusminus == False:
            vmin = min(0, df.values.min())
            vmax = df.values.max()
        else:
            vmin = min(0, df.values.min())
            vmax = df.values.max()
            absmax = max(abs(vmin), vmax)
            vmin = -absmax
            vmax = absmax


        if context == 'ambig':
            sns.heatmap(df, annot=df.round(2), annot_kws={"fontsize": 'x-large'}, ax=location, cmap=colormap, cbar=False, vmin=vmin, vmax=vmax, xticklabels=[])
        else:
            sns.heatmap(df, annot=df.round(2), annot_kws={"fontsize": 'x-large'}, ax=location, cmap=colormap, cbar=False, vmin=vmin, vmax=vmax)
            #location.set_xticklabels(ax.get_xticklabels(), rotation=30)
        #sns.heatmap(df_ambig, annot=df_ambig.round(2), cbar=False, ax=axs[0], cmap=colormap, xticklabels = [], vmin=vmin, vmax=vmax)
        #sns.heatmap(df_disambig, annot=df_disambig.round(2), cbar=False, ax=axs[1], cmap=colormap, vmin=vmin, vmax=vmax)


        #for ax in axs[:2]:
        #    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        #    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        #plt.xticks(rotation=30)

        #plt.show()

        #return fig, axs


    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 7))


    draw_heatmap(tb_amb, 'ambig', ax[0,0], args.cmap_tb); ax[0,0].set_title('Target Bias', size='xx-large')
    draw_heatmap(tb_dis, 'disambig', ax[1, 0], args.cmap_tb)
    draw_heatmap(bamt_amb, 'ambig', ax[0, 1],  args.cmap_bamt); ax[0,1].set_title('Bias Amount', size='xx-large')
    draw_heatmap(bamt_dis, 'disambig', ax[1, 1], args.cmap_bamt)
    draw_heatmap(pb_amb, 'ambig', ax[0, 2], args.cmap_pb); ax[0,2].set_title('Persona Bias', size='xx-large')
    draw_heatmap(pb_dis, 'disambig', ax[1, 2], args.cmap_pb)
    draw_heatmap(bs_amb, 'ambig', ax[0,3], args.cmap_bs, True); ax[0,3].set_title('Bias Score', size='xx-large');
    draw_heatmap(bs_dis, 'disambig', ax[1, 3], args.cmap_bs, True)

    ax[0,0].set_ylabel("Ambiguous", fontsize='xx-large')
    #ax[0,0].spines['left'].set_position(('outward', 20))
    ax[1, 0].set_ylabel("Disambiguated", fontsize='xx-large')
    #ax[1, 0].spines['left'].set_position(('outward', 20))
    #plt.subplots_adjust(left=0.09, bottom=0.12)


    #fig.supxlabel('Domains', size='xx-large')
    #fig.supylabel('Models', size='xx-large')

    fig.tight_layout()

    plt.savefig(os.path.join(args.save_dir, 'result_tables_each.pdf'), dpi=200)

    plt.show()


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

    for cat in args.categories:
        dir = args.result2_dir
        f_name = args.file_name_2.format(args.rp, args.cc)

        file_path = os.path.join(dir, args.model_2, cat, f_name)

        df = pd.read_csv(file_path, index_col=0)

        targets = get_target_list(cat)

        context = 'amb'
        scores = ['polarity', 'amount']

        tb_ti = df[['{}_{}_{}'.format(t, scores[0], context) for t in targets]]
        bamt_ti = df[['{}_{}_{}'.format(t, scores[1], context) for t in targets]]
        tb = df[['TB_{}_{}'.format(scores[0], context)]]
        pb = df[['PB_{}_{}'.format(scores[0], context)]]
        bamt = df[['TB_{}_{}'.format(scores[1], context)]]
        bs = df[['BS_{}'.format('a')]]

        df_figure = pd.concat([tb_ti,pb, bs], axis=1)*100 #  tb,  bamt_ti, bamt



        n_target = len(targets)
        n_persona = len(df_figure)
        if cat != 'SES':
            fig, ax = plt.subplots(figsize=((n_target)/2+1, (n_persona)/3+1))
        else:
            fig, ax = plt.subplots(figsize=((n_target)/2+3, (n_persona)/3+1))
        xticks = targets; xticks.extend(["PB_s", "BS"])

        # tb_ti
        start = 0; end = n_target
        data_tb_ti = df_figure.copy()
        data_tb_ti.iloc[:, end:] = float('nan')

        vmin = np.nanmin(data_tb_ti.values); vmax = np.nanmax(data_tb_ti.values)
        print(vmin, vmax)
        '''
        absmax = max(abs(vmin), vmax)
        vmin = -absmax; vmax = absmax
        print(vmin, vmax)
        '''
        ax = sns.heatmap(data_tb_ti, annot=data_tb_ti.round(0), #annot_kws={"fontsize": 'small'},
                         cmap=args.cmap_tb_i, cbar=False, center=0,
                         xticklabels=xticks,
                         )
                         #vmin=vmin, vmax=vmax)

        '''
        start = end; end = end+1
        data_tb = df_figure.copy()
        data_tb.iloc[:, :start] = float('nan')
        data_tb.iloc[:, end:] = float('nan')
        sns.heatmap(data_tb, annot=data_tb.round(2), cmap=args.cmap_tb, cbar=False)
        '''

        start = end; end = end+1
        data_pb = df_figure.copy()
        data_pb.iloc[:, :start] = float('nan')
        data_pb.iloc[:, end:] = float('nan')
        sns.heatmap(data_pb, annot=data_pb.round(0),
                    #cmap=matplotlib.colors.ListedColormap(['white']), cbar=False,
                    cmap=args.cmap_pb, cbar=False,
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
        '''

        start = end; end = end + 1
        data_bs = df_figure.copy()
        data_bs.iloc[:, :start] = float('nan')
        data_bs.iloc[:, end:] = float('nan')
        sns.heatmap(data_bs, annot=data_bs.round(0),
                    cmap=args.cmap_bs, cbar=False, center=0,
                    xticklabels=xticks)


        plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = False, labeltop=True, rotation=10)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        ax.hlines([1], *ax.get_xlim(), colors='white', linewidth=2)


        if cat == 'Age':
            plt.subplots_adjust(left=0.35, top=0.8)
        if cat == 'Race_ethnicity':
            plt.subplots_adjust(left=0.16, top = 0.79)
        elif cat == 'Religion':
            plt.subplots_adjust(left=0.15, top=0.82)
        elif cat == 'SES':
            plt.subplots_adjust(left=0.4, top=0.92)
        elif cat == 'Sexual_orientation':
            plt.subplots_adjust(left=0.25, top=0.74)

        save_dir = './result_2_TB_ti'
        save_file = '{}_{}.png'.format(cat, context)
        plt.savefig(os.path.join(save_dir, save_file), dpi=200)

        plt.show()

        #break


def result3_stacked_bar(args):
    category = args.category
    result_dir = args.raw_dir
    model = args.model
    f_name = 'refined_p_{}_inst_{}_target_{}*.json'

    if category == 'Race_ethnicity':
        persona = ['Baseline',  'Arab', 'Asian', 'African', 'Caucasian',]
        targets = persona[1:]
    elif category == 'Age':
        persona = ['Baseline', 'kid', 'elder']
        targets = ['nonOld', 'old']
    elif category == 'Religion':
        persona = ['Baseline', 'Protestant', 'Muslim', 'Hindu', 'Buddhist']
        targets = persona[1:]
    categories = ['Baseline'] + [category,] * (len(persona)-1)

    target_category = [category] * len(persona)

    df = pd.DataFrame()

    for cat, p, t in zip(categories, persona, target_category):
        dir_path = os.path.join(result_dir, model, cat)

        dict = {targ: [0, 0, 0] for targ in targets}  # positive / neutral / negative
        n = 0
        for inst in range(args.instruction_k):
            file = f_name.format(p, inst, t)
            file_path = os.path.join(dir_path, file)
            file_name = glob.glob(file_path)[0]

            #print(file_name)

            with open(file_name) as f:
                data = json.load(f)
                f.close()

            #print(len(data))

            for item in data:
                question_polarity = item['question_polarity']  # neg / nonneg
                context_condition = item['context_condition']  # ambig / disambig
                answer_info = item['answer_info']  # {'ans0': [name (option), subcategory], ...}
                stereotyped_group = item['additional_metadata']['stereotyped_groups']  # list
                #if target_category == 'SES':
                #    for idx, s in enumerate(stereotyped_group):
                #        s = s[:-4] + '' + s[-3:]
                #        stereotyped_group[idx] = s
                answer = item['label']
                response = item['response']['refined']
                #if response == 9:
                #    response = unknown_option

                option1 = answer_info['ans0'][1]
                option2 = answer_info['ans1'][1]
                option3 = answer_info['ans2'][1]
                options = [option1, option2, option3]
                #print(options)
                if context_condition != 'ambig':
                    continue

                # 정답
                if (response == 9) or (response == answer):
                    for i in range(3):
                        if response == i:
                            continue
                        if options[i] in targets:
                            dict[options[i]][1] += 1
                else:
                    if options[response] in targets:
                        # 오답 when neg question
                        if question_polarity == 'neg':    # negative question
                            dict[options[response]][2] += 1
                        elif question_polarity == 'nonneg':    # positive question
                            #print("nonneg")
                            dict[options[response]][0] += 1

        print("Persona: {}".format(p))
        print(dict)
        for key in dict:
            print(key)
            total_sum = sum(dict[key])

            percentages = [i/total_sum*100 for i in dict[key]]
            percentages = [round(i, 0) for i in percentages]

            index = '{}-{}'.format(p, key)
            df_temp = pd.DataFrame([percentages], index=[index], columns=['pos', 'neu', 'neg'])
            df = pd.concat([df, df_temp], axis=0)
            print(df)

    fig, ax = plt.subplots(ncols=len(persona), figsize=(len(persona)+1, 4))

    for i in range(len(persona)):
        df.iloc[i * len(dict):(i+1)*len(dict)].plot(kind='bar', stacked=True, ax=ax[i], color=['royalblue', 'silver', 'darkred'], legend=False)
        ax[i].set_xticklabels(dict.keys(), fontsize="large")
        if i != 0:
            ax[i].set_yticklabels([])
        ax[i].set_xlabel(persona[i], fontsize='x-large')

    fig.tight_layout()
    #ax[i].legend(bbox_to_anchor=())
    if category == 'Age':
        def brief_model_name(m):
            if m == 'meta-llama/Llama-2-7b-chat-hf':
                return 'llama-7b'
            if m == 'meta-llama/Llama-2-13b-chat-hf':
                return 'llama-13b'
            if m == 'gpt-3.5-turbo-0613':
                return "GPT3.5"
            if m == 'gpt-4-1106-preview':
                return "GPT4"
            if m == 'meta-llama/Llama-2-70b-chat-hf':
                return 'llama-70b'
        ax[0].set_ylabel(brief_model_name(args.model))
        plt.subplots_adjust(bottom=0.3, top=0.9, left=0.15)
    else:
        plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.suptitle(category)

    save_dir = './result3_stackedbar'
    save_file = '{}_stackedbar_{}.pdf'.format(args.model, category)
    plt.savefig(os.path.join(save_dir, save_file), dpi=200)

    plt.show()


def result4_scatterplot(args):
    file_tb = 'target_bias_{}_rp_{}_cc_{}.csv'
    file_bamt = 'bias_amount_{}_rp_{}_cc_{}.csv'
    file_pb = 'persona_bias_{}_rp_{}_cc_{}.csv'
    file_bs = 'bs_{}_rp_{}_cc_{}.csv'
    file_acc = 'acc_{}_rp_{}_cc_{}.csv'

    def get_df(dir, f_name, rp, cc):
        df_ambig = pd.read_csv(os.path.join(dir, f_name.format('ambig', rp, cc)), index_col=0)
        df_ambig.index = args.cat_names
        df_ambig.columns = args.model_names
        df_disambig = pd.read_csv(os.path.join(dir, f_name.format('disambig', rp, cc)), index_col=0)
        df_disambig.index = args.cat_names
        df_disambig.columns = args.model_names
        return df_ambig.T, df_disambig.T

    tb_amb, tb_dis = get_df(args.save_dir, file_tb, args.rp, args.cc)
    bamt_amb, bamt_dis = get_df(args.save_dir, file_bamt, args.rp, args.cc)
    #pb_amb, pb_dis = get_df(args.save_dir, file_pb, args.rp, args.cc)
    #bs_amb, bs_dis = get_df(args.save_dir, file_bs, args.rp, args.cc)
    acc_amb, acc_dis = get_df(args.save_dir, file_acc, args.rp, args.cc)

    models = tb_amb.index
    domain = tb_amb.columns

    Xs, Ys = [], []

    fig, ax = plt.subplots()
    #markers = ['.', '^', '1', 'x', 's']
    #colors = ['darkred', 'lightcoral', 'y', 'cornflowerblue', 'darkblue']
    markers = ['o'] * 5
    colors = ['royalblue'] * 5
    for d, mark in zip(domain, markers):
        for m, color in zip(models, colors):
            bamt = []
            acc = []
            bamt.append(bamt_amb.at[m, d])
            acc.append(acc_amb.at[m, d])
            Xs.append(bamt_amb.at[m, d])
            Ys.append(acc_amb.at[m, d])
            ax.scatter(bamt, acc, c=color, marker=mark, alpha=0.7)
    #plt.legend()
    #plt.show()

    #fig, ax = plt.subplots()

    for d, mark in zip(domain, markers):
        for m, color in zip(models, colors):
            bamt = []
            acc = []
            bamt.append(bamt_dis.at[m, d])
            acc.append(acc_dis.at[m, d])
            Xs.append(bamt_dis.at[m, d])
            Ys.append(acc_dis.at[m, d])
            ax.scatter(bamt, acc, c=color, marker=mark, alpha=0.7)

    legs = []
    for m, c in zip(models, colors):
        legs.append(matplotlib.patches.Patch(color=c, label=m))
    #ax.legend(handles=legs)
    plt.xlabel("BAmt")
    plt.ylabel("Acc")

    save_dir = './result4_corr'
    save_file = 'corr_bamt_acc.pdf'
    plt.savefig(os.path.join(save_dir, save_file), dpi=200)

    plt.show()

    import scipy.stats as sts
    corr, p = sts.pearsonr(Xs, Ys)
    print(corr, p)


def result5_knn(args):
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

    for cat in args.categories:
        dir = args.result2_dir
        f_name = args.file_name_2.format(args.rp, args.cc)

        file_path = os.path.join(dir, args.model, cat, f_name)

        df = pd.read_csv(file_path, index_col=0)

        targets = get_target_list(cat)

        context = 'amb'
        scores = ['polarity', 'amount']

        tb_ti = df[['{}_{}_{}'.format(t, scores[0], context) for t in targets]]
        #bamt_ti = df[['{}_{}_{}'.format(t, scores[1], context) for t in targets]]
        #tb = df[['TB_{}_{}'.format(scores[0], context)]]
        #pb = df[['PB_{}_{}'.format(scores[0], context)]]
        #bamt = df[['TB_{}_{}'.format(scores[1], context)]]
        #bs = df[['BS_{}'.format('a')]]

        #df_figure = pd.concat([tb_ti,pb, bs], axis=1)*100 #  tb,  bamt_ti, bamt

        n_clusters = 3

        # K means
        kemans = KMeans(n_clusters=n_clusters, n_init=10)
        y = kemans.fit_predict(tb_ti)
        tb_ti['clusters']=y
        # print(tb_ti)

        reduced_data = PCA(n_components=2).fit_transform(tb_ti)
        results = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
        tb_ti.reset_index(inplace=True)
        tb_ti = tb_ti.rename(columns={'index': 'Persona'})
        results = pd.concat([results, tb_ti], axis=1)
        print(results)

        #fig, ax = plt.subplots()
        sns.scatterplot(x="PCA1", y="PCA2", hue='clusters', data=results, s=80)

        for i, (pca1, pca2) in results[['PCA1', 'PCA2']].iterrows():
            plt.text(pca1+0.025, pca2, results.at[i, 'Persona'], color='black', fontsize=15)
        plt.title(cat, fontsize='xx-large')

        save_dir = './result5_pca'
        save_file = 'pca_{}.pdf'.format(cat)
        plt.savefig(os.path.join(save_dir, save_file), dpi=200)
        plt.show()


def result6_barplot_with_line(args):
    category = args.category
    result_dir = args.raw_dir
    model = args.model
    f_name = 'refined_p_{}_inst_{}_target_{}*.json'

    analysis_dir = args.result2_dir
    analysis_f_name = args.file_name_2.format(args.rp, args.cc)

    def get_target_names(category, mode):
        if mode == 'all':
            if category == 'Age':
                targets = ['nonOld', 'old']
            elif category == 'Race_ethnicity':
                targets = ['African', 'African American', 'Arab', 'Asian', 'Black',
                           'Caucasian', 'European', 'Hispanic', 'Jewish', 'Latino',
                           'Middle Eastern', 'Native American', 'Roma', 'South American', 'White']
            elif category == 'Religion':
                targets = ['Atheist', 'Buddhist', 'Catholic', 'Christian', 'Hindu',
                           'Jewish', 'Mormon', 'Muslim', 'Protestant', 'Sikh']
            elif category == 'SES':
                targets=['highSES', 'lowSES']
            elif category == 'Sexual_orientation':
                targets = ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual']
        else:
            if category == 'Race_ethnicity':
                persona = ['Baseline', 'Arab', 'Asian', 'African', 'Caucasian', ]
                targets = persona[1:]
            elif category == 'Age':
                persona = ['Baseline', 'kid', 'elder']
                targets = ['nonOld', 'old']
            elif category == 'Religion':
                persona = ['Baseline', 'Protestant', 'Muslim', 'Hindu', 'Buddhist']
                targets = persona[1:]
        return targets


    persona = ['Baseline']
    targets = get_target_names(category, 'all')
    categories = ['Baseline'] + [category, ] * (len(persona) - 1)

    target_category = [category] * len(persona)

    df = pd.DataFrame()

    for cat, p, t in zip(categories, persona, target_category):
        dir_path = os.path.join(result_dir, model, cat)

        dict = {targ: [0, 0, 0, 0] for targ in targets}  # positive / neutral / negative
        n = 0
        for inst in range(args.instruction_k):
            file = f_name.format(p, inst, t)
            file_path = os.path.join(dir_path, file)
            file_name = glob.glob(file_path)[0]

            # print(file_name)

            with open(file_name) as f:
                data = json.load(f)
                f.close()

            # print(len(data))

            for item in data:
                question_polarity = item['question_polarity']  # neg / nonneg
                context_condition = item['context_condition']  # ambig / disambig
                answer_info = item['answer_info']  # {'ans0': [name (option), subcategory], ...}
                stereotyped_group = item['additional_metadata']['stereotyped_groups']  # list
                # if target_category == 'SES':
                #    for idx, s in enumerate(stereotyped_group):
                #        s = s[:-4] + '' + s[-3:]
                #        stereotyped_group[idx] = s
                answer = item['label']
                response = item['response']['refined']
                # if response == 9:
                #    response = unknown_option

                option1 = answer_info['ans0'][1]
                option2 = answer_info['ans1'][1]
                option3 = answer_info['ans2'][1]
                options = [option1, option2, option3]
                # print(options)
                if context_condition != 'ambig':
                    continue

                for i in range(3):
                    if options[i] in targets:
                        dict[options[i]][3] += 1

                # 정답
                if (response == 9) or (response == answer):
                    for i in range(3):
                        if (response == i) or (response == 9) :
                            continue
                        if options[i] in targets:
                            dict[options[i]][1] += 1
                else:
                    if options[response] in targets:
                        # 오답 when neg question
                        if question_polarity == 'neg':  # negative question
                            #dict[options[response]][2] += 1   # no counter score
                            for i in range(3):
                                if (i == answer) or (response == 9):
                                    continue
                                elif (i == response) :
                                    dict[options[response]][2] += 2
                                else:
                                    dict[options[i]][0] +=1
                        elif question_polarity == 'nonneg':  # positive question
                            # print("nonneg")
                            # dict[options[response]][0] += 1   # no counter score
                            for i in range(3):
                                if (i == answer) or (response == 9):
                                    continue
                                elif (i == response):
                                    dict[options[response]][0] += 2
                                else:
                                    dict[options[i]][2] +=1



        print("Persona: {}".format(p))
        print(dict)
        for key in dict:
            print(key)
            total_sum = sum(dict[key]) - dict[key][3]

            percentages = [i / dict[key][3] for i in dict[key]]
            #percentages = [i / total_sum for i in dict[key]]

            #percentages = [round(i, 0) for i in percentages]

            index = '{}-{}'.format(p, key)
            df_temp = pd.DataFrame([percentages], index=[index], columns=['pos', 'neu', 'neg', 'n'])
            df = pd.concat([df, df_temp], axis=0)
            #print(df)
        print(df)


        # bring tb_ti, bamt
        analysis_path = os.path.join(analysis_dir, model, category, analysis_f_name)
        print(analysis_path)
        analysis_df = pd.read_csv(analysis_path, index_col=0)
        tb_ti = analysis_df.loc[p,['{}_polarity_amb'.format(targ) for targ in targets]]
        bamt_ti = analysis_df.loc[p,['{}_amount_amb'.format(targ) for targ in targets]]
        tb_ti = tb_ti.tolist(); bamt_ti = bamt_ti.tolist()
        df['TB_ti'] = tb_ti
        df['|TB_ti|'] = [abs(x) for x in tb_ti]
        df['BAmt_ti'] = bamt_ti
        print(df)

    fig, ax = plt.subplots(ncols=len(persona), figsize=(len(targets)-4, 4))
    df[['pos', 'neg']].plot(kind='bar', stacked=False, ax=ax, color=['royalblue', 'red'], ylim=(0, 1), legend=False)
    #df[['TB_ti']].plot(kind='line', ax=ax2, color='green', ylim=(-0.1, 1.5))
    ax.set_ylabel("ACC (%)", fontsize='large')

    ax2 = ax.twinx()
    df[['|TB_ti|']].plot(kind='line', linestyle='dashed', ax=ax2, color='mediumorchid', marker='^', ylim=(0, 0.35), legend=False)
    ax2.set_ylabel("|TB_ti|", fontsize='large')

    ax3 = ax.twinx()
    df[['BAmt_ti']].plot(kind='line', ax=ax3, color = 'darkorange', ylim = (0, 1.1), marker='.', legend=False)
    ax3.spines['right'].set_position(('outward', 50))
    ax3.set_ylabel("BAmt_ti", fontsize='large')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines+lines2+lines3, labels+labels2+labels3, loc=0)

    ax.set_xticklabels(targets, rotation=30)
    ax.set_xlabel("Targets", fontsize='x-large')

    def brief_model_name(m):
        if m == 'meta-llama/Llama-2-7b-chat-hf':
            return 'llama-7b'
        if m == 'meta-llama/Llama-2-13b-chat-hf':
            return 'llama-13b'
        if m == 'gpt-3.5-turbo-0613':
            return "GPT3.5"
        if m == 'gpt-4-1106-preview':
            return "GPT4"
        if m == 'meta-llama/Llama-2-70b-chat-hf':
            return 'llama-70b'
    ax.set_title("{}".format(brief_model_name(model)), fontsize='x-large')

    fig.tight_layout()

    plt.savefig('./result6_bar_lines/{}.pdf'.format(brief_model_name(model)), dpi=200)

    plt.show()

    '''
    for i in range(len(persona)):
        df.iloc[i * len(dict):(i + 1) * len(dict)].plot(kind='bar', stacked=True, ax=ax[i],
                                                        color=['royalblue', 'silver', 'darkred'], legend=False)
        ax[i].set_xticklabels(dict.keys(), fontsize="large")
        if i != 0:
            ax[i].set_yticklabels([])
        ax[i].set_xlabel(persona[i], fontsize='x-large')
    
    fig.tight_layout()
    # ax[i].legend(bbox_to_anchor=())
    if category == 'Age':
        def brief_model_name(m):
            if m == 'meta-llama/Llama-2-7b-chat-hf':
                return 'llama-7b'
            if m == 'meta-llama/Llama-2-13b-chat-hf':
                return 'llama-13b'
            if m == 'gpt-3.5-turbo-0613':
                return "GPT3.5"
            if m == 'gpt-4-1106-preview':
                return "GPT4"
            if m == 'meta-llama/Llama-2-70b-chat-hf':
                return 'llama-70b'

        ax[0].set_ylabel(brief_model_name(args.model))
        plt.subplots_adjust(bottom=0.3, top=0.9, left=0.15)
    else:
        plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.suptitle(category)
    '''

    #save_dir = './result3_stackedbar'
    #save_file = '{}_stackedbar_{}.pdf'.format(args.model, category)
    #plt.savefig(os.path.join(save_dir, save_file), dpi=200)

    plt.show()


def collect_tables(args):
    def concat(df_total_amb, df_total_dis, df, score_name, contexts):
        amb = df.loc[:, ['{}_{}'.format(score_name, contexts[0])]].T
        dis = df[['{}_{}'.format(score_name, contexts[1])]].T
        # print(amb.T)

        df_total_amb = pd.concat([df_total_amb, amb], axis=0, ignore_index=True)
        df_total_dis = pd.concat([df_total_dis, dis], axis=0, ignore_index=True)

        return df_total_amb, df_total_dis


    def rename_index_col(df, rows, cols):
        df.index=rows
        df.columns=cols
        return df


    df_tb_amb = pd.DataFrame()
    df_tb_dis = pd.DataFrame()
    df_bamt_amb = pd.DataFrame()
    df_bamt_dis = pd.DataFrame()
    df_pb_amb = pd.DataFrame()
    df_pb_dis = pd.DataFrame()
    df_bs_amb = pd.DataFrame()
    df_bs_dis = pd.DataFrame()
    df_acc_amb = pd.DataFrame()
    df_acc_dis = pd.DataFrame()

    for category in args.categories:
        f_name = args.file_name.format(args.rp, args.cc)
        f_path = os.path.join(args.result_dir, category, f_name)

        df = pd.read_csv(f_path)

        #scores = ['Polarity', 'Amount', 'BS']
        contexts = ['ambig', 'disambig']

        df_tb_amb, df_tb_dis = concat(df_tb_amb, df_tb_dis, df, 'Polarity', contexts)
        df_bamt_amb, df_bamt_dis = concat(df_bamt_amb, df_bamt_dis, df, 'Amount', contexts)
        df_pb_amb, df_pb_dis = concat(df_pb_amb, df_pb_dis, df, 'PB', contexts)
        df_bs_amb, df_bs_dis = concat(df_bs_amb, df_bs_dis, df, 'BS', contexts)
        df_acc_amb, df_acc_dis = concat(df_acc_amb, df_acc_dis, df, 'Acc', contexts)

    df_tb_amb = rename_index_col(df_tb_amb, args.cat_names, args.model_names)
    df_tb_dis = rename_index_col(df_tb_dis, args.cat_names, args.model_names)
    df_bamt_amb = rename_index_col(df_bamt_amb, args.cat_names, args.model_names)
    df_bamt_dis = rename_index_col(df_bamt_dis, args.cat_names, args.model_names)
    df_pb_amb = rename_index_col(df_pb_amb, args.cat_names, args.model_names)
    df_pb_dis = rename_index_col(df_pb_dis, args.cat_names, args.model_names)
    df_bs_amb = rename_index_col(df_bs_amb, args.cat_names, args.model_names)
    df_bs_dis = rename_index_col(df_bs_dis, args.cat_names, args.model_names)
    df_acc_amb = rename_index_col(df_acc_amb, args.cat_names, args.model_names)
    df_acc_dis = rename_index_col(df_acc_dis, args.cat_names, args.model_names)

    df_tb_amb.to_csv(os.path.join(args.save_dir, 'target_bias_ambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_tb_dis.to_csv(os.path.join(args.save_dir, 'target_bias_disambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_bamt_amb.to_csv(os.path.join(args.save_dir, 'bias_amount_ambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_bamt_dis.to_csv(os.path.join(args.save_dir, 'bias_amount_disambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_pb_amb.to_csv(os.path.join(args.save_dir, 'persona_bias_ambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_pb_dis.to_csv(os.path.join(args.save_dir, 'persona_bias_disambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_bs_amb.to_csv(os.path.join(args.save_dir, 'bs_ambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_bs_dis.to_csv(os.path.join(args.save_dir, 'bs_disambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_acc_amb.to_csv(os.path.join(args.save_dir, 'acc_ambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))
    df_acc_dis.to_csv(os.path.join(args.save_dir, 'acc_disambig_rp_{}_cc_{}.csv'.format(args.rp, args.cc)))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dir', type=str, default='./../../results/refined/')
    parser.add_argument('--result_dir', type=str, default='./../total_merged_models/Bias_Score_modifyunknown')
    parser.add_argument('--result2_dir', type=str, default='./../total_merged/Bias_Score_modifyunknown')
    parser.add_argument('--file_name', type=str, default='merged_total_models_rp_{}_cc_{}.csv')
    parser.add_argument('--file_name_2', type=str, default='merged_total_rp_{}_cc_{}.csv')

    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--instruction_k', type=int, default=1)
    parser.add_argument('--category', type=str, default='Religion')

    parser.add_argument('--save_dir', type=str, default='./result_tables')
    parser.add_argument('--rp', type=int, default=2)
    parser.add_argument('--cc', type=int, default=1)

    parser.add_argument('--categories', type=list, default=['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation'])
    parser.add_argument('--cat_names', type=list, default=['Age', 'Race', 'Religion', 'SES', 'SexualO'])
    parser.add_argument('--model_names', type=list, default=['llama7b', 'llama13b', 'llama70b', 'gpt-3.5', 'gpt-4'])

    parser.add_argument('--cmap_tb_i', type=str, default='bwr_r')
    parser.add_argument('--cmap_tb', type=str, default='Purples')
    parser.add_argument('--cmap_bamt', type=str, default='Oranges')
    parser.add_argument('--cmap_pb', type=str, default='bone_r')
    parser.add_argument('--cmap_bs', type=str, default='vlag')


    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    #collect_tables(args)
    #main(args)
    result1_main_heatmap(args)
    #result2(args)
    #for model in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'gpt-3.5-turbo-0613', 'gpt-4-1106-preview', 'meta-llama/Llama-2-70b-chat-hf']:
    #    for cat in ['Age', 'Religion', 'Race_ethnicity']:
    #        args.model = model
    #        args.category = cat
    #        result3_stacked_bar(args)

    #result4_scatterplot(args)
    #result5_knn(args)

    #result6_barplot_with_line(args)