import pathlib

import numpy as np
import seaborn as sns

from horaha import PROJECT_ROOT
from horaha.utils import compute_posterior_means


def plot_identifiers(ax, Z,
                     identifiers=None, fontsize=10, horizontalalignment='left'):
    k, n = Z.shape
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xoffset = sum(map(abs, xlim)) / 100.0
    yoffset = sum(map(abs, ylim)) / 100.0
    if identifiers is None:
        identifiers = {i: str(i + 1) for i in range(n)}
    for i, s in identifiers.items():
        ax.text(Z[0, i] + xoffset, Z[1, i] + yoffset,
                s, fontsize=fontsize, horizontalalignment=horizontalalignment)

    return ax


def plot_links(ax, Z, Y, color='k', linestyle='dashed'):
    n = Y.shape[0]
    IJLINK = 1
    JILINK = 2
    links = {}
    for i in range(n):
        for j in range(i + 1, n):
            yij = IJLINK * (Y[i, j] == 1)
            yji = JILINK * (Y[j, i] == 1)
            links[(i, j)] = yij + yji

    def _plot_arrow(i, j, b0=0.05, b1=0.05):
        x0 = Z[0, i]
        y0 = Z[1, i]
        x1 = Z[0, j]
        y1 = Z[1, j]
        np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        x0_ = x0 + b0 * (x1 - x0)
        y0_ = y0 + b0 * (y1 - y0)
        x1_ = x1 - b1 * (x1 - x0)
        y1_ = y1 - b1 * (y1 - y0)
        dx = x1_ - x0_
        dy = y1_ - y0_
        ax.arrow(x0_, y0_, dx, dy,
                 length_includes_head=True,
                 facecolor=color, edgecolor=color,
                 head_width=0.3, head_length=0.5,)

    for (i, j), kind in links.items():
        if kind == 0:
            continue
        elif kind == IJLINK:
            _plot_arrow(i, j)
        elif kind == JILINK:
            _plot_arrow(j, i)
        elif kind == (IJLINK + JILINK):
            ax.plot(Z[0, [i, j]], Z[1, [i, j]],
                    color=color, linestyle=linestyle)
    return ax


def plot_points(ax, Z, **kwargs):
    '''Plot MLE Z, a (k, n) array'''
    ax.scatter(Z[0, :], Z[1, :], **kwargs)
    return ax


def plot_samples(ax, Z, size=2, clip=0):
    '''Plot MCMC samples Z, a list of (k, n) array'''
    k, n = Z[0].shape

    # compute mean positions of each agent and get colors by angle
    means = compute_posterior_means(Z)
    angles = np.arctan2(means[1, :], means[0, :])
    order = np.argsort(angles)
    c = sns.color_palette(palette='viridis', n_colors=n)

    # plot posterior samples
    for Z_k in Z:
        plot_points(ax, Z_k[:, order], s=size, c=c)

    # manually set limits
    tmp = np.array(Z).transpose([1, 0, 2]).reshape(2, -1)
    lo_x1, lo_x2 = np.percentile(tmp, clip / 2, axis=1)
    hi_x1, hi_x2 = np.percentile(tmp, 100 - clip / 2, axis=1)
    ax.set_xlim(left=lo_x1, right=hi_x1)
    ax.set_ylim(bottom=lo_x2, top=hi_x2)

    return ax


def set_misc(ax):
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def plot_trace(ax, θ):
    '''Plot trace of draws of θ, a list of float'''
    K = len(θ)

    # plot
    ax.plot(np.arange(K), θ, 'k-')

    # add labels, etc
    ax.set_xlabel('sample')

    return ax


def savefig(fig, name):
    figdir = pathlib.Path(PROJECT_ROOT).joinpath('paper', 'figures')
    for ext in ['.png', '.eps']:
        fig.savefig(str(figdir / (name + ext)),
                    bbox_inches='tight', pad_inches=0)
