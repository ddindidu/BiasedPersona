import os, glob, argparse
import json
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from plot_diagonal_heatmap import result2, result2_for_case
from plot_bar_with_lines import result6_barplot_with_line


def result1_main_heatmap(args):
    def get_df(dir, f_name, context, rp, cc):
        models = ['llama7b', 'llama13b', 'llama70b', 'gpt-3.5', 'gpt-4']
        domains = ['Age', 'Race_ethnicity', 'Religion', 'SES', 'Sexual_orientation']
        tb = pd.DataFrame()
        bamt = pd.DataFrame()
        pb = pd.DataFrame()
        bs = pd.DataFrame()

        cont = 'a' if context == 'ambig' else 'd'

        for domain in domains:
            source_dir = os.path.join(dir, domain)
            source_f_name = f_name.format(domain, context, rp, cc)
            source_path = os.path.join(source_dir, source_f_name)

            df  = pd.read_csv(source_path, index_col=0)
            df.index = models

            tb = pd.concat([tb, df['TB']], axis=1)
            bamt = pd.concat([bamt, df['BAmt']], axis=1)
            pb = pd.concat([pb, df['PB']], axis=1)
            bs = pd.concat([bs, df['BS_{}'.format(cont)]], axis=1)

        short_domains = ['Age', 'Race', 'Religion', 'SES', 'SexualO']
        tb.columns = short_domains
        bamt.columns = short_domains
        pb.columns = short_domains
        bs.columns = short_domains

        return tb, bamt, pb, bs

    tb_amb, bamt_amb, pb_amb, bs_amb = get_df(args.source_dir, args.source_file, 'ambig', args.rp, args.cc)
    tb_dis, bamt_dis, pb_dis, bs_dis = get_df(args.source_dir, args.source_file, 'disambig', args.rp, args.cc)


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


    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 5))


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

    plt.savefig(os.path.join(args.save_dir, 'result_tables_each.png'), dpi=200)
    plt.savefig(os.path.join(args.save_dir, 'result_tables_each.pdf'), dpi=200)

    plt.show()



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

        p = persona[i] if persona[i] != 'Baseline' else 'Default'
        ax[i].set_xlabel(p, fontsize='x-large')


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
        ax[0].set_ylabel(brief_model_name(args.model), fontsize='x-large')
        #plt.subplots_adjust(bottom=0.3, top=0.9, left=0.15)
    #else:
        #plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.suptitle(category if category != 'Race_ethnicity' else 'Race/Ethnicity', fontsize='x-large')

    fig.tight_layout()

    save_dir = './result3_stackedbar'
    save_file = '{}_stackedbar_{}.pdf'.format(args.model, category)
    plt.savefig(os.path.join(save_dir, save_file), dpi=200, transparent=True)

    plt.show()


def result4_scatterplot(args):
    '''
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
    '''
    def get_df(dir, f_name, domain, context, rp, cc):
        f_path = os.path.join(dir, domain,
                              f_name.format(domain, context, rp, cc))
        df = pd.read_csv(f_path, index_col=0)

        return df


    models = args.model_names
    domain = args.categories

    Xs, Ys = [], []

    fig, ax = plt.subplots()

    #markers = ['.', '^', '1', 'x', 's']
    markers = ['x', 'o']
    colors = ['darkred', 'lightcoral', 'y', 'cornflowerblue', 'darkblue']

    bamt_list, acc_list = [], []
    for d in domain:
        for cont, mark in zip(['ambig', 'disambig'], markers):
            df = get_df(args.source_dir, args.source_file, d, cont, args.rp, args.cc)

            for row, color in zip(df.iterrows(), colors):
                #print(row)
                bamt = row[1]['BAmt']
                acc = row[1]['Acc_{}'.format('a' if cont == 'ambig' else 'd')]
                Xs.append(bamt)
                Ys.append(acc)
                ax.scatter(bamt, acc, c=color, marker=mark, alpha=0.7)

    '''
    for d, mark in zip(domain, markers):
        for m, color in zip(models, colors):
            bamt = []
            acc = []
            bamt.append(bamt_amb.at[m, d])
            acc.append(acc_amb.at[m, d])
            Xs.append(bamt_amb.at[m, d])
            Ys.append(acc_amb.at[m, d])
            ax.scatter(bamt, acc, c=color, marker=mark, alpha=0.7)

    for d, mark in zip(domain, markers):
        for m, color in zip(models, colors):
            bamt = []
            acc = []
            bamt.append(bamt_dis.at[m, d])
            acc.append(acc_dis.at[m, d])
            Xs.append(bamt_dis.at[m, d])
            Ys.append(acc_dis.at[m, d])
            ax.scatter(bamt, acc, c=color, marker=mark, alpha=0.7)
    '''
    import scipy.stats as sts
    corr, p = sts.pearsonr(Xs, Ys)
    print(corr, p)

    ax.line


    legs = []
    for m, c in zip(models, colors):
        legs.append(matplotlib.patches.Patch(color=c, label=m))
    ax.legend(handles=legs, fontsize='large')
    plt.xlabel("BAmt", fontsize='large')
    plt.ylabel("Acc", fontsize='large')

    plt.tight_layout()

    save_dir = './result4_corr'
    save_file = 'corr_bamt_acc.pdf'
    plt.savefig(os.path.join(save_dir, save_file), dpi=200)

    plt.show()




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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dir', type=str, default='./../../results/refined/')
    parser.add_argument('--source_dir', type=str, default='./../total_score_model_merged/')
    parser.add_argument('--source_dir2', type=str, default='./../total_score')
    parser.add_argument('--source_file', type=str, default='{}_{}_rp_{}_cc_{}.csv')   # domain_context_rp_cc
    parser.add_argument('--file_name_2', type=str, default='aver_{}_{}_rp_{}_cc_{}.csv')    # domain_context_rp_cc

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    #parser.add_argument('--model', type=str, default='meta-llama/Llama-2-70b-chat-hf')
    parser.add_argument('--instruction_k', type=int, default=5)
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

    #main(args)

    #result1_main_heatmap(args)

    #result2(args)
    #result2_for_case(args)

    for model in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'gpt-3.5-turbo-0613', 'gpt-4-1106-preview', 'meta-llama/Llama-2-70b-chat-hf']:
        for cat in ['Age', 'Religion', 'Race_ethnicity']:
            args.model = model
            args.category = cat

            if 'gpt-3.5' in args.model:
                args.instruction_k=5
            elif 'llama' in args.model:
                args.instruction_k=3
            else:
                args.instruction_k=1

            result3_stacked_bar(args)

    #result4_scatterplot(args)
    #result5_knn(args)

    #result6_barplot_with_line(args)