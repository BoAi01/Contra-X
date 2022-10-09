import pandas as pd
from pandarallel import pandarallel

from utils import process, extract_style

pandarallel.initialize()

authors = ['human', 'ctrl', 'fair_wmt19', 'fair_wmt20', 'gpt1', 'gpt2_large', 'gpt2_medium', 'gpt2_pytorch',
           'gpt2_small', 'gpt2_xl', 'gpt3', 'grover_base', 'grover_large', 'grover_mega', 'pplm_distil', 'pplm_gpt2',
           'transfo_xl', 'xlm', 'xlnet_base', 'xlnet_large']

splits = ['test', 'train', 'valid']

# for split_name in splits:

#     df = pd.read_csv(f'turing_dataset/AA/{split_name}.csv')
#     df['From'] = df['label'].apply(lambda x: authors.index(x))
#     df['train'] = 1 if split_name == 'train' else 0
#     df['author'] = df['label']
#     df['content'] = df['Generation']

#     df['content_tfidf'] = df['content'].apply(lambda x: process(x))
#     df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
#         "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
#         "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
#         "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
#         "f_e_10", "f_e_11", "richness"]] = df['content'].apply(lambda x: extract_style(x))

#     df.to_csv(f'turing_dataset/AA/{split_name}_processed.csv')

# for split_name in splits:
#     for author in authors:
#         if author != 'human':

#             df = pd.read_csv(f'turing_dataset/TT_{author}/{split_name}.csv')
#             df['From'] = df['label'].apply(lambda x: authors.index(x) if authors.index(x) == 0 else 1)
#             df['train'] = 1 if split_name == 'train' else 0
#             df['author'] = df['label']
#             df['content'] = df['Generation']

#             df['content_tfidf'] = df['content'].apply(lambda x: process(x))
#             df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
#                 "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
#                 "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
#                 "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
#                 "f_e_10", "f_e_11", "richness"]] = df['content'].apply(lambda x: extract_style(x))

#             df.to_csv(f'turing_dataset/TT_{author}/{split_name}_processed.csv')
# infile = open(f'turing_dataset/AA/{split_name}.csv', 'r')
# outfile = open(f'turing_dataset/AA/{split_name}_processed.csv', 'w')

# line = infile.readline()
# while line != '':
# update 20220208: consider all train(1) - test(0) - valid(2)
final_df = pd.read_csv(f'datasets/TuringBench/AA/train.csv')
for rd, split_name in enumerate(splits):
    df = pd.read_csv(f'datasets/TuringBench/AA/{split_name}.csv')
    df['From'] = df['label'].parallel_apply(lambda x: authors.index(x))
    if split_name == 'train':
        df['train'] = 1
    elif split_name == 'test':
        df['train'] = 0
    else:
        assert split_name == 'valid'
        df['train'] = 2
    df['author'] = df['label']
    df['content'] = df['Generation']

    df['content_tfidf'] = df['content'].parallel_apply(lambda x: process(x))
    df[["avg_len", "len_text", "len_words", "num_short_w", "per_digit", "per_cap", "f_a", "f_b", "f_c", "f_d",
        "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m", "f_n", "f_o", "f_p", "f_q", "f_r", "f_s",
        "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7",
        "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4", "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9",
        "f_e_10", "f_e_11", "richness"]] = df['content'].parallel_apply(lambda x: extract_style(x))
    if rd == 0:
        final_df = df
    else:
        final_df = pd.concat([final_df, df], axis=0)
final_df.to_csv(f'datasets/turing_ori_0208.csv')
