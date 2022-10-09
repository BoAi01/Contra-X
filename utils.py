# general
import os

# visualization
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib.collections import QuadMesh
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# evaluation
from sklearn.model_selection import train_test_split

# dataset
from prepare_dataset import dataset_path


dataset_path = "datasets"
dataset_file_name = {
    "imdb62": 'full_imdb62.csv',
    "blog": 'full_blog.csv',
    "turing": "turing_ori_0208.csv"
}


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy();
        dic['color'] = 'g';
        lis_kwa.append(dic);
        dic = text_kwargs.copy();
        dic['color'] = 'r';
        lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col


def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
                                 lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim
    plt.show()


def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
                                    fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    # data
    if (not columns):
        from string import ascii_uppercase
        columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize = [9, 9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values,
                                 pred_val_axis=pred_val_axis)


def fil_sent(sent):
    """
    Filter stopwords
    """
    stop_words = set(stopwords.words('english'))
    filtered_sentence = ' '.join([w for w in sent.split() if not w in stop_words])
    return filtered_sentence


def process(sent):
    """
    Apply stemming
    """
    sent = str(sent)
    ps = PorterStemmer()
    return fil_sent(' '.join([ps.stem(str(x).lower()) for x in word_tokenize(sent)]))


def extract_style(text):
    """
    Extracting stylometric features of a text
    """

    text = str(text)
    len_text = len(text)
    len_words = len(text.split())
    avg_len = np.mean([len(t) for t in text.split()])
    num_short_w = len([t for t in text.split() if len(t) < 3])
    per_digit = sum(t.isdigit() for t in text) / len(text)
    per_cap = sum(1 for t in text if t.isupper()) / len(text)
    f_a = sum(1 for t in text if t.lower() == "a") / len(text)
    f_b = sum(1 for t in text if t.lower() == "b") / len(text)
    f_c = sum(1 for t in text if t.lower() == "c") / len(text)
    f_d = sum(1 for t in text if t.lower() == "d") / len(text)
    f_e = sum(1 for t in text if t.lower() == "e") / len(text)
    f_f = sum(1 for t in text if t.lower() == "f") / len(text)
    f_g = sum(1 for t in text if t.lower() == "g") / len(text)
    f_h = sum(1 for t in text if t.lower() == "h") / len(text)
    f_i = sum(1 for t in text if t.lower() == "i") / len(text)
    f_j = sum(1 for t in text if t.lower() == "j") / len(text)
    f_k = sum(1 for t in text if t.lower() == "k") / len(text)
    f_l = sum(1 for t in text if t.lower() == "l") / len(text)
    f_m = sum(1 for t in text if t.lower() == "m") / len(text)
    f_n = sum(1 for t in text if t.lower() == "n") / len(text)
    f_o = sum(1 for t in text if t.lower() == "o") / len(text)
    f_p = sum(1 for t in text if t.lower() == "p") / len(text)
    f_q = sum(1 for t in text if t.lower() == "q") / len(text)
    f_r = sum(1 for t in text if t.lower() == "r") / len(text)
    f_s = sum(1 for t in text if t.lower() == "s") / len(text)
    f_t = sum(1 for t in text if t.lower() == "t") / len(text)
    f_u = sum(1 for t in text if t.lower() == "u") / len(text)
    f_v = sum(1 for t in text if t.lower() == "v") / len(text)
    f_w = sum(1 for t in text if t.lower() == "w") / len(text)
    f_x = sum(1 for t in text if t.lower() == "x") / len(text)
    f_y = sum(1 for t in text if t.lower() == "y") / len(text)
    f_z = sum(1 for t in text if t.lower() == "z") / len(text)
    f_1 = sum(1 for t in text if t.lower() == "1") / len(text)
    f_2 = sum(1 for t in text if t.lower() == "2") / len(text)
    f_3 = sum(1 for t in text if t.lower() == "3") / len(text)
    f_4 = sum(1 for t in text if t.lower() == "4") / len(text)
    f_5 = sum(1 for t in text if t.lower() == "5") / len(text)
    f_6 = sum(1 for t in text if t.lower() == "6") / len(text)
    f_7 = sum(1 for t in text if t.lower() == "7") / len(text)
    f_8 = sum(1 for t in text if t.lower() == "8") / len(text)
    f_9 = sum(1 for t in text if t.lower() == "9") / len(text)
    f_0 = sum(1 for t in text if t.lower() == "0") / len(text)
    f_e_0 = sum(1 for t in text if t.lower() == "!") / len(text)
    f_e_1 = sum(1 for t in text if t.lower() == "-") / len(text)
    f_e_2 = sum(1 for t in text if t.lower() == ":") / len(text)
    f_e_3 = sum(1 for t in text if t.lower() == "?") / len(text)
    f_e_4 = sum(1 for t in text if t.lower() == ".") / len(text)
    f_e_5 = sum(1 for t in text if t.lower() == ",") / len(text)
    f_e_6 = sum(1 for t in text if t.lower() == ";") / len(text)
    f_e_7 = sum(1 for t in text if t.lower() == "'") / len(text)
    f_e_8 = sum(1 for t in text if t.lower() == "/") / len(text)
    f_e_9 = sum(1 for t in text if t.lower() == "(") / len(text)
    f_e_10 = sum(1 for t in text if t.lower() == ")") / len(text)
    f_e_11 = sum(1 for t in text if t.lower() == "&") / len(text)
    richness = len(list(set(text.split()))) / len(text.split())

    return pd.Series(
        [avg_len, len_text, len_words, num_short_w, per_digit, per_cap, f_a, f_b, f_c, f_d, f_e, f_f, f_g, f_h, f_i,
         f_j, f_k, f_l, f_m, f_n, f_o, f_p, f_q, f_r, f_s, f_t, f_u, f_v, f_w, f_x, f_y, f_z, f_0, f_1, f_2, f_3,
         f_4, f_5, f_6, f_7, f_8, f_9, f_e_0, f_e_1, f_e_2, f_e_3, f_e_4, f_e_5, f_e_6, f_e_7, f_e_8, f_e_9, f_e_10,
         f_e_11, richness])


def build_train_test(df, source, limit, per_author=None, seed=None):
    # Select top N senders and build Train and Test
    list_spk = list(pd.DataFrame(df['From'].value_counts().iloc[:limit]).reset_index()['index'])
    sub_df = df[df['From'].isin(list_spk)]

    if per_author is not None:
        raise NotImplementedError()

    if source == 'turing':
        sub_df = sub_df[
            [
                'From', 'content', 'content_tfidf', "avg_len", "len_text", "len_words", "num_short_w", "per_digit",
                "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m",
                "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1",
                "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4",
                "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11", "richness", "train"
            ]
        ]
    else:
        sub_df = sub_df[
            [
                'From', 'content', 'content_tfidf', "avg_len", "len_text", "len_words", "num_short_w", "per_digit",
                "per_cap", "f_a", "f_b", "f_c", "f_d", "f_e", "f_f", "f_g", "f_h", "f_i", "f_j", "f_k", "f_l", "f_m",
                "f_n", "f_o", "f_p", "f_q", "f_r", "f_s", "f_t", "f_u", "f_v", "f_w", "f_x", "f_y", "f_z", "f_0", "f_1",
                "f_2", "f_3", "f_4", "f_5", "f_6", "f_7", "f_8", "f_9", "f_e_0", "f_e_1", "f_e_2", "f_e_3", "f_e_4",
                "f_e_5", "f_e_6", "f_e_7", "f_e_8", "f_e_9", "f_e_10", "f_e_11", "richness"
            ]
        ]
    sub_df = sub_df.dropna()

    print("Number of texts : ", len(sub_df))

    dict_nlp_enron = {}
    k = 0

    for val in np.unique(sub_df.From):
        dict_nlp_enron[val] = k
        k += 1

    sub_df['Target'] = sub_df['From'].apply(lambda x: dict_nlp_enron[x])

    if source == 'turing':
        perc = 0.5
        print("Percentage: " + str(perc))
        full_train = sub_df[sub_df["train"] == 1]
        nlp_train = full_train[['content', 'Target']]

        full_test = sub_df[sub_df["train"] == 0]
        test_dict = full_test[['content', 'Target']]

        full_valid = sub_df[sub_df["train"] == 2]
        val_dict = full_valid[['content', 'Target']]

        shrinked_train = nlp_train
        shrinked_test = test_dict
        shrinked_val = val_dict
        for l in range(20):
            part_train = nlp_train[nlp_train["Target"] == l]
            part_train = part_train[:int(len(part_train) * perc)]
            part_test = test_dict[test_dict["Target"] == l]
            part_test = part_test[:int(len(part_test) * perc)]
            part_val = val_dict[val_dict["Target"] == l]
            part_val = part_val[:int(len(part_val) * perc)]
            if l == 0:
                shrinked_train = part_train
                shrinked_test = part_test
                shrinked_val = part_val
            else:
                shrinked_train = pd.concat([shrinked_train, part_train], axis=0)
                shrinked_test = pd.concat([shrinked_test, part_test], axis=0)
                shrinked_val = pd.concat([shrinked_val, part_val], axis=0)

        return shrinked_train, shrinked_test, shrinked_val

    if 'blog' in source or 'imdb62' in source:
        perc = 0.75
        print("seed: " + str(seed))

        if seed is None:
            seed = 0

        ind = train_test_split(sub_df[['content', 'Target']], test_size=0.2, stratify=sub_df['Target'],
                               random_state=seed)
        ind_train = list(ind[0].index)
        nlp_train = sub_df.loc[ind_train]

        val_test_sub_df = ind[1]
        ind2 = train_test_split(val_test_sub_df[['content', 'Target']], test_size=0.5,
                                stratify=val_test_sub_df['Target'], random_state=seed)
        ind_val = list(ind2[0].index)
        ind_test = list(ind2[1].index)
        val_dict = val_test_sub_df.loc[ind_val]
        test_dict = val_test_sub_df.loc[ind_test]

        if 'blog' in source:
            shrinked_train = nlp_train
            shrinked_test = test_dict
            shrinked_val = val_dict
            for l in range(50):
                part_train = nlp_train[nlp_train["Target"] == l]
                part_train = part_train[:int(len(part_train) * perc)]
                part_test = test_dict[test_dict["Target"] == l]
                part_test = part_test[:int(len(part_test) * perc)]
                part_val = val_dict[val_dict["Target"] == l]
                part_val = part_val[:int(len(part_val) * perc)]
                if l == 0:
                    shrinked_train = part_train
                    shrinked_test = part_test
                    shrinked_val = part_val
                else:
                    shrinked_train = pd.concat([shrinked_train, part_train], axis=0)
                    shrinked_test = pd.concat([shrinked_test, part_test], axis=0)
                    shrinked_val = pd.concat([shrinked_val, part_val], axis=0)

            return shrinked_train, shrinked_test, shrinked_val

        return nlp_train, val_dict, test_dict

    ind = train_test_split(sub_df[['content', 'Target']], test_size=0.2, stratify=sub_df['Target'], random_state=seed)
    ind_train = list(ind[0].index)
    ind_test = list(ind[1].index)
    nlp_train = sub_df.loc[ind_train]
    test_dict = sub_df.loc[ind_test]

    return nlp_train, test_dict


def is_name_in_email(name, email):
    """
    Removing emails from Enron where name is in email
    """

    if str(name).lower() in str(email).lower():
        return 1
    else:
        return 0


def load_dataset_dataframe(source):
    print("Loading and processing dataframe")

    # dataset_path = "datasets"
    # dataset_file_name = {
    #     "imdb62": 'full_imdb62.csv',
    #     "blog": 'full_blog.csv',
    #     "turing": "turing_ori_0208.csv"
    # }

    df = None
    if source == "imdb62":
        df = pd.read_csv(os.path.join(dataset_path, dataset_file_name[source]), index_col=0)
    elif source == "blog":
        df = pd.read_csv(os.path.join(dataset_path, dataset_file_name[source]))
    else:
        df = pd.read_csv(os.path.join(dataset_path, dataset_file_name[source]))
        df.sort_values(by=['train', 'From'], inplace=True, ascending=[False, True])

    return df


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(ckpt_dir, cp_name, model):
    """
    Create directory /Checkpoint under exp_data_path and save encoder as cp_name
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    saving_model_path = os.path.join(ckpt_dir, cp_name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # convert to non-parallel form
    torch.save(model.state_dict(), saving_model_path)
    print(f'Model saved: {saving_model_path}')


def load_model_dic(model, ckpt_path, verbose=True, strict=True):
    """
    Load weights to model and take care of weight parallelism
    """
    assert os.path.exists(ckpt_path), f"trained model {ckpt_path} does not exist"

    try:
        model.load_state_dict(torch.load(ckpt_path), strict=strict)
    except:
        state_dict = torch.load(ckpt_path)
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=strict)
    if verbose:
        print(f'Model loaded: {ckpt_path}')

    return model

