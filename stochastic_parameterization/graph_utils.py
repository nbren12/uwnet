import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as color_map
from sklearn.metrics import auc


def draw_horizontal_barplot(heights, names, label='Counts', title=None):
    y_pos = np.arange(len(names))
    plt.barh(y_pos, heights, align='center', alpha=0.5)
    plt.yticks(y_pos, names)
    plt.ylabel(label)
    plt.xlabel(label)
    if title:
        plt.title(title)
    plt.show()


def draw_barplot(heights, names, y_label='Counts', title=None,
                 xticks_rotation=0,
                 ylim_min=None,
                 ylim_max=None,
                 fontsize=12,
                 xlabels_max_chars=float('inf'),
                 x_label='',
                 width=0.8,
                 yline=None,
                 xline=None,
                 figsize=(6, 4),
                 bottom=None,
                 show=True,
                 save_to_filepath=None):
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=figsize)
    if yline:
        ax.axhline(y=yline, color='k', linewidth=0.75)
    if xline:
        ax.axvline(x=xline, color='k', linewidth=0.75)
    plt.bar(y_pos, heights, align='center', alpha=0.5, width=width)
    plt.xticks(y_pos, names, rotation=xticks_rotation, fontsize=fontsize)
    if ylim_min is None:
        ylim_min = float(min(heights))
    if ylim_max is None:
        ylim_max = float(max(heights))
    plt.ylim((ylim_min, ylim_max))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if title:
        plt.title(title)
    if bottom is not None:
        plt.subplots_adjust(bottom=bottom)
    if save_to_filepath:
        plt.savefig(save_to_filepath)
    if show:
        plt.show()
    plt.close()


def draw_barplot_multi(heights_list,
                       names,
                       y_label='Counts',
                       title=None,
                       xticks_rotation=0,
                       ylim_min=None,
                       ylim_max=None,
                       fontsize=12,
                       xlabels_max_chars=float('inf'),
                       width=None,
                       show=True,
                       figsize=None,
                       legend_labels=None,
                       legend_loc="best",
                       save_to_filepath=None,
                       text_x_y=(),
                       text=None,
                       text_kwargs=None):
    n_plot = len(heights_list)
    n_marks = len(names)
    if not width:
        # width = figsize[0] / n_plot / n_marks
        width = 0.7 / len(heights_list)
    x_pos = np.arange(n_marks)
    min_y = float('inf')
    max_y = -float('inf')
    colors = iter(color_map.rainbow(np.linspace(0, 1, n_plot)))
    plt.figure(figsize=figsize)
    for idx, heights in enumerate(heights_list):
        legend_label = None
        if legend_labels:
            legend_label = legend_labels[idx]
        plt.bar(x_pos + idx * width, heights, width=width, color=next(colors),
                label=legend_label)
        min_y = min(min_y, min(heights))
        max_y = max(max_y, max(heights))
    plt.xticks(x_pos + ((n_plot - 1) * width / 2), names, rotation=xticks_rotation,
               fontsize=fontsize)
    if ylim_min is None:
        ylim_min = min_y
    if ylim_max is None:
        ylim_max = max_y + (max_y * .05)
    plt.ylim((ylim_min, ylim_max))
    plt.ylabel(y_label)
    if legend_labels:
        plt.legend(loc=legend_loc)
    if title:
        plt.title(title)
    plt.ticklabel_format(axis='y', style='plain')
    if text:
        plt.text(*text_x_y, text, **text_kwargs)
    if save_to_filepath:
        plt.savefig(save_to_filepath)
    if show:
        plt.show()
    return plt


def draw_barplot_multi_iter(heights_list, names, n_per_iter, kwargs):
    n = len(names)
    idx_start = 0
    idx_end = min(idx_start + n_per_iter, n)
    while idx_start < n:
        hl = [list(x)[idx_start:idx_end] for x in heights_list]
        draw_barplot_multi(hl, list(names)[idx_start:idx_end],
                           **kwargs)
        idx_start = idx_end
        idx_end = min(idx_start + n_per_iter, n)


def draw_histogram(values, bins=40, x_min=None, x_max=None,
                   x_label='', y_label='Counts', title='',
                   figsize=None,
                   show=True, save_to_filepath=None):
    if x_max is None:
        x_max = max(values)
    if x_min is None:
        x_min = min(values)
    if figsize is not None:
        plt.figure(figsize=figsize)
    n_, bins, patches = plt.hist(values, bins=bins)
    plt.plot(bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axis([x_min, x_max, 0, max(n_)])
    plt.grid(True)
    if save_to_filepath:
        plt.savefig(save_to_filepath)
    if show:
        plt.show()
    plt.close()


def draw_histogram_normalized_by_count(values, bins=40, x_max=None,
                                       x_label='', y_label='Counts', title=''):
    if x_max is None:
        x_max = max(values)
    bin_vals = list(np.arange(0, x_max, x_max / (bins - 1)))
    bin_vals.append(x_max)
    hist, bin_edges = np.histogram(values, bins=bin_vals)
    values_norm = list(hist / len(values))
    plt.plot(values_norm)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axis([min(values), x_max, 0, max(values_norm)])
    plt.grid(True)
    plt.show()


def draw_elb_device_histogram(df, attr, x_max, count_threshold=0, bins=40,
                              normalize_by_count=False):
    value_counts = df[attr].value_counts()
    for key in df[attr].unique():
        count = value_counts[key]
        if count > count_threshold:
            values = df[(df[attr] == key) & (df.pct_agreement > 0)
                        ].pct_agreement.values.astype(np.float32) * 100
            if (normalize_by_count):
                draw_histogram_normalized_by_count(
                    values, bins=bins, x_max=x_max,
                    title='{}: {} (count={})'.format(attr, key, count))
            else:
                draw_histogram(values, bins=bins, x_max=x_max,
                               title='{}: {} (count={})'.format(attr, key, count))


def draw_elb_device_histogram_all_in_one_plot(df, attr, max_pct_agreement,
                                              count_threshold=0, bins=40,
                                              normalize_by_count=False):
    x_max = max_pct_agreement
    y_max = 0
    title = '{}'.format(attr)
    value_counts = df[attr].value_counts()
    for key in df[attr].unique():
        count = value_counts[key]
        if count > count_threshold:
            values = df[(df[attr] == key) & (df.pct_agreement > 0)
                        ].pct_agreement.values.astype(np.float32) * 100
            if x_max is None:
                x_max = max(values)
            bin_vals = list(np.arange(0, x_max, x_max / (bins - 1)))
            bin_vals.append(x_max)
            hist, bin_edges = np.histogram(values, bins=bin_vals)
            values_norm = list(hist / len(values))
            if max(values_norm) > y_max:
                y_max = max(values_norm)
            plt.plot(values_norm, label='{} (count={})'.format(key, count))
    plt.xlabel('pct_agreement')
    plt.ylabel('normalized counts (percents)')
    plt.title(title)
    plt.axis([0, x_max, 0, y_max])
    plt.grid(True)
    plt.legend()
    plt.show()


def setup_and_show_roc_curve(xlabel, ylabel, title):
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curve(roc_df, show=True, include_auc=True,
                   color='darkorange',
                   legend_label='',
                   legend_label_max_chars=40,
                   xlabel='False Positive Rate',
                   ylabel='True Positive Rate',
                   title='ROC Curve',
                   figsize=(10, 10)):
    if not set(['fpr', 'tpr']).issubset(list(roc_df.columns)):
        raise Exception("Columns 'fpr' and 'tpr' must be in roc_df.")
    if show:
        plt.figure(figsize=figsize)
    if include_auc:
        auc_ = auc(roc_df['fpr'], roc_df['tpr'])
        legend_label_max_chars = legend_label_max_chars - 10
        legend_label = '{} (auc={:0.2f})'.format(legend_label, auc_)
    plt.plot(roc_df['fpr'], roc_df['tpr'], color=color,
             label=legend_label)
    if show:
        setup_and_show_roc_curve(xlabel, ylabel, title)


def create_plotting_names(df, name_columns_list, start_index, end_index,
                          sep=' - '):
    names = [str(x)
             for x in list(df[name_columns_list[0]][start_index:end_index])]
    for name in name_columns_list[1:]:
        names = ['{}{}{}'.format(x[0], sep, x[1]) for x in zip(names,
                                                               list(df[name][start_index:end_index]))]
    return names


def plot_roc_curve_df(roc_df, n_plot=10, include_auc=True, sort_by_auc=True,
                      xlabel='False Positive Rate',
                      ylabel='True Positive Rate',
                      title='ROC Curve',
                      legend_columns_list=['segment_id'],
                      legend_columns_sep=' - ',
                      legend_label_max_chars=40,
                      figsize=(10, 10)):
    missing_columns = list(
        set(['roc'] + legend_columns_list) - set(roc_df.columns))
    if missing_columns:
        raise Exception('column(s) {} missing from roc_df'.format(
            missing_columns))
    if sort_by_auc:
        roc_df['auc'] = roc_df.apply(lambda x: auc(x['roc']['fpr'],
                                                   x['roc']['tpr']), axis=1)
        roc_df = roc_df.sort_values('auc', ascending=False).reset_index(
            drop=True)
    n_segs = len(roc_df)
    for i in range(0, n_segs, n_plot):
        plt.figure(figsize=figsize)
        colors = iter(color_map.rainbow(np.linspace(0, 1, n_plot)))
        for j in range(i, min(i + n_plot, n_segs)):
            legend_label = create_plotting_names(roc_df, legend_columns_list,
                                                 j, j + 1,
                                                 sep=legend_columns_sep)[0]
            df = roc_df.loc[j, 'roc']
            plot_roc_curve(df, show=False, include_auc=include_auc,
                           color=next(colors), legend_label=legend_label,
                           legend_label_max_chars=legend_label_max_chars)
        setup_and_show_roc_curve(xlabel, ylabel, title)


def plot_roc_curve_dict(roc_df_dict, n_plot=10, include_auc=True,
                        xlabel='False Positive Rate',
                        ylabel='True Positive Rate',
                        title='ROC Curve'):
    for idx, item in enumerate(roc_df_dict.items()):
        if (idx % n_plot) == 0:
            plt.figure()
            colors = iter(color_map.rainbow(np.linspace(0, 1, n_plot)))
        key = item[0]
        roc_df = item[1]
        plot_roc_curve(roc_df, show=False, include_auc=include_auc,
                       color=next(colors), legend_label=str(key))
        if ((idx % n_plot) == (n_plot - 1)) or (idx == (len(roc_df_dict) - 1)):
            setup_and_show_roc_curve(xlabel, ylabel, title)


def plot_auc_df(auc_df, n_plot=20, sort_values=True,
                name_columns_list=['name'],
                name_columns_sep=' - ',
                xlabels_max_chars=30,
                figsize=(8, 6)):
    missing_columns = list(
        set(['auc'] + name_columns_list) - set(auc_df.columns))
    if missing_columns:
        raise Exception('column(s) {} missing from auc_df'.format(
            missing_columns))
    if sort_values:
        auc_df = auc_df.sort_values(by='auc', ascending=False)
    n_segs = len(auc_df)
    ylim_max = math.ceil(max(auc_df['auc']))
    for i in range(0, n_segs, n_plot):
        j = min(i + n_plot, n_segs)
        names = create_plotting_names(auc_df, name_columns_list, i, j,
                                      sep=name_columns_sep)
        plt.figure(figsize=figsize)
        draw_barplot(auc_df['auc'][i:j],
                     names,
                     y_label='AUC',
                     title='AUC by segment_id',
                     xticks_rotation=90,
                     ylim_max=ylim_max,
                     xlabels_max_chars=xlabels_max_chars,
                     show=False)
        ylim_min_i = min(auc_df['auc'][i:j])
        if ylim_min_i < 0.5:
            plt.axhline(y=0.5, color='k')
        plt.show()


def draw_scatterplot(x, y, xlabel='', ylabel='', title='', color='b',
                     xticks=None, yticks=None, figsize=None,
                     xline=None, yline=None,
                     pt_labels=[], pt_label_offset=(5, -2),
                     text=None, text_offset=(0.1, 0.1),
                     slope=None, intercept=None,
                     save_to_filepath=None,
                     show=True, additional_x=None, additional_y=None):
    if type(x) != list:
        x = list(x)
    if type(y) != list:
        y = list(y)
    fig, ax = plt.subplots(figsize=figsize)
    plt.scatter(x, y, color=color)
    if yline:
        ax.axhline(y=yline, color='k', linewidth=0.75)
    if xline:
        ax.axvline(x=xline, color='k', linewidth=0.75)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if len(pt_labels) > 0:
        for i, txt in enumerate(pt_labels):
            ax.annotate(txt, xy=(x[i], y[i]), xytext=pt_label_offset,
                        textcoords='offset points')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if slope is not None and intercept is not None:
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')
    if additional_x is not None and additional_y is not None:
        plt.plot(additional_x, additional_y, '--')
    if text:
        plt.text(text_offset[0], text_offset[1], text, transform=ax.transAxes,
                 bbox=dict(facecolor='none', pad=4.0, linewidth=0.75))
    if save_to_filepath:
        plt.savefig(save_to_filepath)
    if show:
        plt.show()
    plt.close()