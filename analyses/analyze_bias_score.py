import os, argparse
import glob
import json
import pandas as pd
from scoring import scoring_overall, get_overall_acc
from scoring import scoring_bbq, get_bs_ambig, get_bs_disambig
from scoring import scoring_kbbq, get_diff_bias_ambig, get_diff_bias_disambig
from scoring import scoring_ours_ambig, scoring_ours_disambig, dataframe_scoring_by_level

import sys
sys.path.append('./../test/')
from persona import call_persona_list
from utils import dir_checker


def make_dataframe(persona_list, target_list):
    #persona_list.sort()
    #target_list.sort()
    d = dict()
    for t in target_list:
        d[t] = [0]*len(persona_list)
    df = pd.DataFrame(data=d, index=persona_list)
    return df


def get_target_list(args, target_category):
    if target_category in ['Religion', 'Sexual_orientation']:
        target_list = call_persona_list(args.source_dir, 'persona_list.csv', target_category)['persona_list']
    elif target_category in ['SES', 'Race_ethnicity', 'Age']:
        target_list = call_persona_list(args.source_dir, 'persona_list.csv', target_category)['Subcategory']
    elif target_category in ['Nationality']:
        if args.target_level == 'subcategory':
            target_list = call_persona_list(args.source_dir, 'persona_list.csv', target_category)['Subcategory']
        else:
            target_list = call_persona_list(args.source_dir, 'persona_list.csv', target_category)['persona_list']
    return target_list


def get_target_level(target_category, target_level, stereotyped_name, anti_stereotyped_name, stereotyped_subcategory,
                     anti_stereotyped_subcategory):
    if target_category in ['Religion', 'Sexual_orientation']:
        stereotyped_item = stereotyped_name
        anti_stereotyped_item = anti_stereotyped_name
    elif target_category in ['SES', 'Race_ethnicity', 'Age']:
        stereotyped_item = stereotyped_subcategory
        anti_stereotyped_item = anti_stereotyped_subcategory
    elif target_category in ['Nationality']:
        if target_level == 'subcategory':
            stereotyped_item = stereotyped_subcategory
            anti_stereotyped_item = anti_stereotyped_subcategory
        else:
            stereotyped_item = stereotyped_name
            anti_stereotyped_item = anti_stereotyped_name
    return stereotyped_item, anti_stereotyped_item


def main(args):
    persona_category = args.persona_category
    target_category = args.target_category

    persona_list = call_persona_list(args.source_dir, 'persona_list.csv', persona_category)['persona_list']
    target_list = get_target_list(args, target_category)

    result_dir = os.path.join(args.result_dir, args.model, persona_category)
    file_name = '*{}*.json'.format(target_category)

    file_list = glob.glob(os.path.join(result_dir, file_name))

    print(persona_list)
    print(len(persona_list))
    print(len(file_list))

    reward_penalty = args.rp
    counter_reward_penalty = args.cc
    instruction_k = args.instruction_k

    persona_list.sort()
    target_list.sort()
    file_list.sort()

    save_file_name_dict  = {'overall': list(), 'ambig': list(), 'disambig': list()}

    for inst_no in range(args.instruction_k):
        df_bias_score_overall = pd.DataFrame()
        df_score_ambig = make_dataframe(persona_list, target_list)
        df_cnt_ambig = make_dataframe(persona_list, target_list)
        df_score_disambig = make_dataframe(persona_list, target_list)
        df_cnt_disambig = make_dataframe(persona_list, target_list)

        for p_no, persona in enumerate(persona_list):
            file_idx = p_no * instruction_k + inst_no
            f_name = file_list[file_idx]
            print(persona, inst_no, file_idx, f_name)
            with open(f_name, 'r') as f:
                data = json.load(f)
                f.close()

            n_overall = {
                'n_ambig': 0, 'n_disambig': 0, 'n_ambig_correct': 0, 'n_disambig_correct': 0
            }
            n_ambig, n_ambig_correct, n_disambig_biased_ans, n_disambig_nonUnknown_ans = 0, 0, 0, 0
            kbbq_n_ambig, kbbq_n_ambig_biased, kbbq_n_ambig_counterbiased = 0,0,0
            kbbq_n_disambig_neg, kbbq_n_disambig_nonneg, kbbq_n_disambig_neg_correct, kbbq_n_disambig_nonneg_correct = 0,0,0,0

            for item in data:
                question_polarity = item['question_polarity']   # neg / nonneg
                context_condition = item['context_condition']   # ambig / disambig
                answer_info = item['answer_info']   # {'ans0': [name (option), subcategory], ...}
                stereotyped_group = item['additional_metadata']['stereotyped_groups']   # list
                if target_category == 'SES':
                    for idx, s in enumerate(stereotyped_group):
                        s = s[:-4] + '' + s[-3:]
                        stereotyped_group[idx] = s
                answer = item['label']
                response = item['response']['refined']

                unknown_option = None
                stereotyped_option, stereotyped_name, stereotyped_subcategory = None, None, None
                anti_stereotyped_option, anti_stereotyped_name, anti_stereotyped_subcategory = None, None, None

                for ans_idx in range(3):
                    ans_info = answer_info['ans{}'.format(ans_idx)]
                    option_name, option_subcat = ans_info[0], ans_info[1]

                    if target_category == 'Race_ethnicity':
                        if ("F-" in option_subcat) or ("M-" in option_subcat):
                            option_subcat = option_subcat[2:]

                    # unknown option
                    if option_subcat == "unknown":
                        unknown_option = ans_idx
                    # stereotyped_answer
                    elif (option_name in stereotyped_group) or (option_subcat in stereotyped_group):
                        stereotyped_option = ans_idx
                        stereotyped_name = option_name
                        stereotyped_subcategory = option_subcat
                    # anti-/counter-stereotyped_answer
                    else:
                        anti_stereotyped_option = ans_idx
                        anti_stereotyped_name = option_name
                        anti_stereotyped_subcategory = option_subcat

                '''
                if response == 9:
                    print("AI CANNOT ANSWER")
                    if context_condition == 'ambig':
                        if question_polarity == 'neg':
                            response = stereotyped_option
                        else:
                            response = anti_stereotyped_option
                    else:
                        response = unknown_option
                    response = unknown_option
                '''

                # Accuracy
                n_overall = scoring_overall(context_condition, answer, response, n_overall)

                # BBQ Bias Score
                n_ambig, n_ambig_correct, n_disambig_nonUnknown_ans, n_disambig_biased_ans = scoring_bbq(context_condition, question_polarity, answer, response,
                                                                                                         stereotyped_option, anti_stereotyped_option,
                                                                                                         n_ambig, n_ambig_correct, n_disambig_biased_ans, n_disambig_nonUnknown_ans)

                # KBBQ Bias Score
                kbbq_n_ambig, kbbq_n_ambig_biased, kbbq_n_ambig_counterbiased, \
                    kbbq_n_disambig_neg, kbbq_n_disambig_nonneg, kbbq_n_disambig_neg_correct, kbbq_n_disambig_nonneg_correct \
                    = scoring_kbbq(context_condition, question_polarity, answer, response,
                                   stereotyped_option, anti_stereotyped_option,
                                   kbbq_n_ambig, kbbq_n_ambig_biased, kbbq_n_ambig_counterbiased,
                                   kbbq_n_disambig_neg, kbbq_n_disambig_nonneg, kbbq_n_disambig_neg_correct,
                                   kbbq_n_disambig_nonneg_correct)

                # mine
                stereotyped_item, anti_stereotyped_item = get_target_level(target_category, args.target_level,
                                                                           stereotyped_name, anti_stereotyped_name,
                                                                           stereotyped_subcategory, anti_stereotyped_subcategory)

                if context_condition == 'ambig':
                    #'''
                    df_cnt_ambig, df_score_ambig = \
                        scoring_ours_ambig(df_cnt_ambig, df_score_ambig,
                                          persona, stereotyped_item, anti_stereotyped_item,
                                          answer, response,
                                          unknown_option, stereotyped_option, anti_stereotyped_option,
                                          question_polarity, context_condition,
                                           reward_penalty, counter_reward_penalty)
                    #'''
                    #df_cnt_ambig, df_score_ambig = dataframe_scoring_by_level(df_cnt_ambig, df_score_ambig,
                    #                                                          persona, stereotyped_item, anti_stereotyped_item,
                    #                                                          answer, response, unknown_option, stereotyped_option, anti_stereotyped_option,
                    #                                                          question_polarity, context_condition)
                else:   # disambig
                    df_cnt_disambig, df_score_disambig = \
                        scoring_ours_disambig(args, df_cnt_disambig, df_score_disambig,
                                            persona, stereotyped_item,
                                            anti_stereotyped_item,
                                            answer, response,
                                            unknown_option, stereotyped_option, anti_stereotyped_option,
                                            question_polarity, context_condition,
                                            reward_penalty, counter_reward_penalty
                                            )


            acc_ambig, acc_disambig = get_overall_acc(n_overall)
            score_disambig = get_bs_disambig(n_disambig_biased_ans, n_disambig_nonUnknown_ans)
            score_ambig = get_bs_ambig(n_ambig, n_ambig_correct, score_disambig)
            kbbq_diff_bias_ambig = get_diff_bias_ambig(kbbq_n_ambig, kbbq_n_ambig_biased, kbbq_n_ambig_counterbiased)
            kbbq_diff_bias_disambig = get_diff_bias_disambig(kbbq_n_disambig_neg, kbbq_n_disambig_neg_correct, kbbq_n_disambig_nonneg, kbbq_n_disambig_nonneg_correct)

            origin_score = {
                'Persona': persona,
                'BS_d': score_disambig, 'BS_a': score_ambig,
                'Diff_Bias_d': kbbq_diff_bias_disambig, 'Diff_Bias_a': kbbq_diff_bias_ambig,
                'Acc_d': acc_ambig, 'Acc_a': acc_disambig
            }
            df_origin_score = pd.DataFrame.from_dict([origin_score])
            df_bias_score_overall = pd.concat([df_bias_score_overall, df_origin_score], ignore_index=True, axis=0)

        #print(df_score_ambig)
        #print(df_score_ambig/df_cnt_ambig)
        df_result_ambig = df_score_ambig/df_cnt_ambig
        df_result_disambig = df_score_disambig/df_cnt_disambig

        save_file_name = save_file(args, inst_no, df_bias_score_overall, df_result_ambig, df_result_disambig)
        save_file_name_dict = accumulate_save_file_name(save_file_name_dict, save_file_name)

    # merge all dataset
    if args.instruction_k > 1:
        df_overall, df_ambig, df_disambig = average_scores(save_file_name_dict)
        _ = save_file(args, None, df_overall, df_ambig, df_disambig)


def average_scores(save_file_name_dict):
    def average_df(file_list):
        n = len(file_list)
        total_df = pd.DataFrame()
        for f_no, f_name in enumerate(file_list):
            df_temp = pd.read_csv(f_name, index_col=0)

            if f_no == 0:
                total_df = df_temp
            else:
                total_df = total_df.add(other=df_temp, fill_value=0)
        return total_df.div(n)
    # overall score
    f_list = save_file_name_dict['overall']
    df_overall = average_df(f_list)
    # ambig score
    f_list = save_file_name_dict['ambig']
    df_ambig = average_df(f_list)
    # disambig score
    f_list = save_file_name_dict['disambig']
    df_disambig = average_df(f_list)
    return df_overall, df_ambig, df_disambig



def accumulate_save_file_name(dict, save_file_name):
    for key in dict:
        dict[key].append(save_file_name[key])
    return dict

def save_file(args, inst_no, df_overall, df_result_ambig, df_result_disambig):
    persona_category, target_category = args.persona_category, args.target_category
    reward_penalty, counter_reward_penalty = args.rp, args.cc

    score_dir = os.path.join(args.output_dir, args.model, args.persona_category)
    dir_checker(score_dir)

    file_name_root = ""

    if inst_no is not None:
        file_name_root += 'inst_{}_'.format(inst_no)
    else:
        file_name_root += 'aver_'

    if args.target_level == 'subcategory':
        file_name_root += '{}2{}'.format(persona_category, target_category)
    else:  # 'name'
        file_name_root += '{}_{}2{}_{}'.format(persona_category, target_category, args.target_level)
    f_name_overall = '_overall_score'
    f_name_ambig = '_ambig_score_rp_{}_cc_{}'.format(reward_penalty, counter_reward_penalty)
    f_name_disambig = '_disambig_score_rp_{}_cc_{}'.format(reward_penalty, counter_reward_penalty)

    file_name_overall = file_name_root+f_name_overall+'.csv'
    file_path_overall = os.path.join(score_dir, file_name_overall)
    if inst_no is not None:
        df_overall.set_index('Persona', inplace=True)
    df_overall.to_csv(file_path_overall)
    print("FILE SAVED: {}".format(file_path_overall))

    file_name_ambig = file_name_root+f_name_ambig+'.csv'
    file_path_ambig = os.path.join(score_dir, file_name_ambig)
    df_result_ambig.to_csv(file_path_ambig)
    print("FILE SAVED: {}".format(file_path_ambig))

    file_name_disambig = file_name_root+f_name_disambig+'.csv'
    file_path_disambig = os.path.join(score_dir, file_name_disambig)
    df_result_disambig.to_csv(file_path_disambig)
    print("FILE SAVED: {}".format(file_path_disambig))

    save_file_name = {
        'overall': file_path_overall,
        'ambig': file_path_ambig,
        'disambig': file_path_disambig,
    }
    return save_file_name



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_dir', type=str, default='./../source')
    parser.add_argument('--result_dir', type=str, default='./../results/refined')
    parser.add_argument('--output_dir', type=str, default='./Bias_Score')
    parser.add_argument('--new_score_deno', type=int, default=0)

    #parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')
    #parser.add_argument('--instruction_k', type=int, default=5)
    parser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--instruction_k', type=int, default=1)

    parser.add_argument('--persona_category', type=str, default='Baseline')
    parser.add_argument('--target_category', type=str, default='Race_ethnicity')
    parser.add_argument('--target_level', type=str, default='subcategory')

    parser.add_argument('--rp', type=int, default=2)    # reward and penalty score  : {1, 2}
    parser.add_argument('--cc', type=int, default=1)    # counter-reward and counter-penalty score: {0, 1}

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    #print(args)

    points = [(2, 1), (1, 1), (1, 0)]

    fields = []
    if args.model == 'gpt-3.5-turbo-0613':
        fields = ['Sexual_orientation', 'Age', 'Race_ethnicity', 'Religion', 'SES']
    else:
        fields = ['Age', 'Race_ethnicity', 'Religion', 'Sexual_orientation', 'SES']

    for point in points:
        args.rp = point[0]
        args.cc = point[1]
        for field in fields:
            args.target_category = field
            persona_categories = ['Baseline']
            #if args.model == 'gpt-3.5-turbo-0613':
            persona_categories.append(args.target_category)
            #main(args)



            for p in persona_categories:
                args.persona_category = p

                print(args)
                main(args)

