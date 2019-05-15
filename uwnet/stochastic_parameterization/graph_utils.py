import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as color_map
from sklearn.metrics import auc
import string
import matplotlib.colors as colors


article_style = {'axes.titlesize': 'medium', 'axes.labelsize': 'small'}
plt.style.use('tableau-colorblind10')


def nbsubplots(nrows=1, ncols=1, w=None, h=1.0, aspect=1.0, **kwargs):
    """Make a set of axes with fixed aspect ratio"""
    from matplotlib import pyplot as plt

    if w is not None:
        h = w * aspect
    else:
        w = h / aspect

    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), **kwargs)


def figlabel(*args, fig=None, **kwargs):
    """Put label in figure coords"""
    if fig is None:
        fig = plt.gcf()
    plt.text(*args, transform=fig.transFigure, **kwargs)


def loghist(x,
            logy=True,
            gaussian_comparison=True,
            ax=None,
            lower_percentile=1e-5,
            upper_percentile=100 - 1e-5,
            label='Sample',
            colors=('k', 'g'),
            cstyle={}):
    """
    Plot log histogram of given samples with normal comparison using
    kernel density estimation
    """
    from scipy.stats import gaussian_kde, norm
    from numpy import percentile

    if ax is None:
        ax = plt.axes()

    p = gaussian_kde(x)

    npts = 100

    p1 = percentile(x, lower_percentile)
    p2 = percentile(x, upper_percentile)
    xx = np.linspace(p1, p2, npts)

    if logy:
        y = np.log(p(xx))
    else:
        y = p(xx)

    ax.plot(xx, y, label=label)

    if gaussian_comparison:
        mles = norm.fit(x)
        gpdf = norm.pdf(xx, *mles)
        if logy:
            ax.plot(xx, np.log(gpdf), label='Gauss', **cstyle)
        else:
            ax.plot(xx, gpdf, label='Gauss', **cstyle)

    ax.set_xlim([p1, p2])


def test_loghist():
    from numpy.random import normal

    x = normal(size=1000)
    loghist(x)
    plt.legend()
    plt.show()


def plot2d(x, y, z, ax=None, cmap='RdGy', norm=None, **kw):
    """ Plot dataset using NonUniformImage class
    Parameters
    ----------
    x : (nx,)
    y : (ny,)
    z : (nx,nz)

    """
    from matplotlib.image import NonUniformImage
    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(111)

    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    im = NonUniformImage(ax,
                         interpolation='bilinear',
                         extent=xlim + ylim,
                         cmap=cmap)

    if norm is not None:
        im.set_norm(norm)

    im.set_data(x, y, z, **kw)
    ax.images.append(im)
    #plt.colorbar(im)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    def update(z):
        return im.set_data(x, y, z, **kw)

    return im, update


def test_plot2d():
    x = np.arange(10)
    y = np.arange(20)

    z = x[None, :]**2 + y[:, None]**2
    plot2d(x, y, z)
    plt.show()


def func_plot(df,
              func,
              w=1,
              aspect=1.0,
              figsize=None,
              layout=(-1, 3),
              sharex=False,
              sharey=False,
              **kwargs):
    """Plot every column in dataframe with func(series, ax=ax, **kwargs)"""
    ncols = df.shape[1]

    q, r = divmod(ncols, layout[-1])

    nrows = q
    if r > 0:
        nrows += 1

    # Adjust figsize
    if not figsize:
        figsize = (w * layout[-1], w * aspect * nrows)
    fig, axs = plt.subplots(nrows,
                            layout[1],
                            figsize=figsize,
                            sharex=sharex,
                            sharey=sharey)
    lax = axs.ravel().tolist()
    for i in range(ncols):
        ser = df.iloc[:, i]
        ax = lax.pop(0)
        ax.text(.1,
                .8,
                df.columns[i],
                bbox=dict(fc='white'),
                transform=ax.transAxes)
        func(ser, ax=ax, **kwargs)

    for ax in lax:
        fig.delaxes(ax)


def pgram(x, ax=None):
    from scipy.signal import welch
    f, Pxx = welch(x.values)
    if not ax:
        ax = plt.gca()

    ax.loglog(f, Pxx)
    ax.grid()
    ax.autoscale(True, tight=True)


def labelled_bar(x, ax=None, pad=200, **kw):
    """A bar chart for a pandas series x with labelling
    x.plot(kind='hist') labels the xaxis only of the plots, and it is nice to
    label the actual bars directly.
    """
    locs = np.arange((len(x)))

    if not ax:
        fig, ax = plt.subplots()

    rects = ax.bar(locs, x, **kw)
    ax.set_xticks(locs + .5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    def autolabel(rects, labels, ax=None, pad=pad):

        for rect, lab in zip(rects, labels):
            lab = str(lab)
            height = rect.get_height() * (1 if rect.get_y() >= 0 else -1)

            kw = {'ha': 'center'}
            if height < 0:
                kw['va'] = 'top'
                height -= ax.transScale.inverted().transform((0, pad))[1]
            else:
                kw['va'] = 'bottom'
                height += ax.transScale.inverted().transform((0, pad))[1]

            ax.text(rect.get_x() + rect.get_width() / 2., height, lab, **kw)

    autolabel(rects, x.index, ax=ax, pad=pad)
    return rects, ax


class LogP1(colors.Normalize):
    """Logarithmic norm for variables from [0, infty]"""

    def __init__(self, data=None, base=10, **kwargs):
        colors.Normalize.__init__(self, **kwargs)
        if data is not None:
            base = np.percentile(data, 90) + 1
        self.base = base

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.log(1 + value) / np.log(self.base))


def plotiter(l,
             ncol=3,
             yield_axis=False,
             figsize=None,
             w=1,
             aspect=1.0,
             tight_layout=True,
             label=None,
             label_dict={},
             labeltype='alpha',
             title=False,
             sharex=False,
             sharey=False,
             **kwargs):
    """Return a generator wrapping an iterator with matplotlib subplots
    This function is used in a similar manner to seaborns FacetGrid class, but
    is designed to work with standard python data structures.
    Parameters
    ----------
    l: seq
        An iterator whose which will be yielded
    ncol: int, optional
        The maximum number of columns
    yield_axis: bool, optional
        if True, a matplotlib axes object is also yielded
    Yields
    ------
    obj:
       an element of input iterator
    ax: matplotlib.pyplot.axes, optional
       axes object is returned if yield_axis=True
    """

    # Label_dict defaults:
    label_kwargs = dict(labeltype=labeltype, loc=(-.05, 1.1), title=title)
    label_kwargs.update(label_dict)

    l = list(l)
    n = len(l)

    ncol = min(n, ncol)

    nrow = np.ceil(n / ncol)

    if figsize is None:
        figsize = (w * ncol, aspect * w * nrow)

    plt.figure(figsize=figsize, **kwargs)

    for i in range(n):

        subplot_kwargs = {}
        if i > 0:
            if sharex:
                subplot_kwargs['sharex'] = ax
            if sharey:
                subplot_kwargs['sharey'] = ax


        ax = plt.axes(plt.subplot(nrow, ncol, i + 1, **subplot_kwargs))

        if labeltype is not None:
            if label_kwargs['labeltype'] == 'alpha':
                label = string.ascii_uppercase[i]
            elif label_kwargs['labeltype'] == 'iter':
                label = l[i]


            if label_kwargs.get('title', True):
                ax.set_title(label)
            else:
                args = label_kwargs['loc'] + (label,)
                ax.text(*args, transform=ax.transAxes,
                        fontdict=dict(weight='bold', size='x-large'))

        if yield_axis:
            yield l[i], ax
        else:
            yield l[i]

    if tight_layout:
        plt.tight_layout()

    return


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


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
    plt.xticks(
        x_pos + ((n_plot - 1) * width / 2), names, rotation=xticks_rotation,
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
                   label=None,
                   pdf=False,
                   show=True, save_to_filepath=None):
    n_, bins, patches = plt.hist(values, bins=bins, density=pdf, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
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
