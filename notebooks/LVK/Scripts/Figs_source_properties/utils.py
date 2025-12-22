import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import gridspec
import matplotlib.pyplot as plt
import corner
from seaborn.palettes import SEABORN_PALETTES
from pesummary.io import read
colorblind = SEABORN_PALETTES["colorblind"]

result_file = "../../Results/posterior_samples.h5"

read_result = read(result_file)
samples = read_result.samples_dict


samples_map = dict(
    NRSur7dq4=samples["C00:NRSur7dq4"],
    IMRPhenomXPHM=samples["C00:IMRPhenomXPHM-SpinTaylor"],
    IMRPhenomTPHM=samples["C00:IMRPhenomTPHM"],
    SEOBNRv5PHM=samples["C00:SEOBNRv5PHM"],
    IMRPhenomXO4a=samples["C00:IMRPhenomXO4a"],
    combined=samples["C00:Mixed"]
)

color_dict = {
    "NRSur7dq4": colorblind[2], #colorblind[1],
    "IMRPhenomXPHM": colorblind[1],
    "IMRPhenomTPHM": colorblind[4],
    "SEOBNRv5PHM": colorblind[0],
    "IMRPhenomXO4a": colorblind[7],
    "combined": "#7c5295", #2e0e4b",
    "combined_td": "#2e0e4b",
    "combined_fd": "#2e0e4b",
}

linestyle_dict = {
    "NRSur7dq4": "-.",
    "IMRPhenomXPHM": "-",
    "IMRPhenomTPHM": "-",
    "SEOBNRv5PHM": "-",
    "IMRPhenomXO4a": "-",
    "combined": "-",
    "combined_td": "--",
    "combined_fd": "-."
}

alpha_dict = {
    "NRSur7dq4": 0.3,
    "IMRPhenomXPHM": 0.3,
    "IMRPhenomTPHM": 0.3,
    "SEOBNRv5PHM": 0.3,
    "IMRPhenomXO4a": 0.3,
    "combined": 0.3,
    "combined_td": 0.3,
    "combined_fd": 0.3,
}

label_dict = {
    "NRSur7dq4": "NRSur",
    "IMRPhenomXPHM": "XPHM",
    "IMRPhenomTPHM": "TPHM",
    "SEOBNRv5PHM": "v5PHM",
    "IMRPhenomXO4a": "XO4a",
    "combined": "Combined",
    "combined_td": "Combined TD",
    "combined_fd": "Combined FD"
}

linewidth_dict = {
    "NRSur7dq4": 1.0,
    "IMRPhenomXPHM": 1.0,
    "IMRPhenomTPHM": 1.0,
    "SEOBNRv5PHM": 1.0,
    "IMRPhenomXO4a": 1.0,
    "combined": 2.5,
    "combined_td": 2.5,
    "combined_fd": 2.5
}


def _add_percentile_lines(
    percentiles, ax, ptype="part", color=None, lw=None, ls=None, zorder=None,
    swap_axis=False
):
    function = ax.axvline
    kwargs = {"ymax": 0.05}
    if swap_axis:
        function = ax.axhline
        kwargs = {"xmax": 0.05}
    for pp in percentiles:
        if ptype is None:
            pass
        elif ptype == "part":
            function(pp, color=color, lw=lw, ls=ls, zorder=zorder, **kwargs)
        elif ptype == "full":
            function(pp, color=color, lw=lw, ls=ls, zorder=zorder)


def add_samples_to_fig(
    xx, xxlims, label, alpha, ls, c, ax, zorder=-10, weights=1., no_fill=False,
    swap_axis=False, ptype="part", linewidth=2, line_alpha=1.0,
    kde_kernel=gaussian_kde, kde_kwargs={}, hpd=False
):
    xlow, xhigh = xxlims
    xsmooth = np.linspace(xlow, xhigh, 1000)

    lw = linewidth
    kde = kde_kernel(xx, **kde_kwargs)
    if hpd:
        from pesummary.utils.credible_interval import hpd_two_sided_credible_interval
        percentiles, _ = hpd_two_sided_credible_interval(
            xx, 90, kde=kde_kernel, kde_kwargs=kde_kwargs,
            xlow=xlow, xhigh=xhigh
        )
    else:
        try:
            percentiles = xx.confidence_interval([5, 95])
        except AttributeError:
            percentiles = [np.percentile(xx, i) for i in [5, 95]]
    if not swap_axis:
        ax.plot(
            xsmooth, weights * kde(xsmooth), color=c, lw=lw, ls=ls, label=label,
            zorder=zorder, alpha=line_alpha
        )
        _add_percentile_lines(
            percentiles, ax, ptype=ptype, color=c, lw=2.0, ls=ls,
            zorder=zorder
        )
    else:
        ax.plot(
            weights * kde(xsmooth), xsmooth, color=c, lw=lw, ls=ls, label=label,
            zorder=zorder
        )
        _add_percentile_lines(
            percentiles, ax, ptype=ptype, color=c, lw=2.0, ls=ls, zorder=zorder,
            swap_axis=True
        )
    if not no_fill and not swap_axis:
        ax.fill_between(
            xsmooth, 0, weights * kde(xsmooth), color=c, alpha=alpha, zorder=zorder
        )
    elif not no_fill:
        ax.fill_betweenx(
            xsmooth, 0, weights * kde(xsmooth), color=c, alpha=alpha, zorder=zorder
        )


def triangle_plot_2d_axes(
    xbounds, ybounds, figsize=(8, 8),
    width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.0, hspace=0.0,
    grid=False):
    """Initialize the axes for a 2d triangle plot.
    """
    high1d = 1.05

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios,
                           wspace=wspace, hspace=hspace)
    ax3 = plt.subplot(gs[2])
    ax1 = plt.subplot(gs[0])
    ax4 = plt.subplot(gs[3])

    ax1.minorticks_on()
    ax3.minorticks_on()
    ax4.minorticks_on()

    if grid:
        ax1.grid(which='major', ls='-')
        ax1.grid(which='minor', ls=':')
        ax3.grid(which='major', ls='-')
        ax3.grid(which='minor', ls=':')
        ax4.grid(which='major', ls='-')
        ax4.grid(which='minor', ls=':')

    # Get rid of tick labels
    ax1.xaxis.set_ticklabels([])
    ax4.yaxis.set_ticklabels([])

    # Use consistent x-axis and y-axis bounds in all 3 plots
    ax1.set_ylim(0, high1d)
    ax1.set_xlim(xbounds[0], xbounds[1])
    ax3.set_xlim(xbounds[0], xbounds[1])
    ax3.set_ylim(ybounds[0], ybounds[1])
    ax4.set_xlim(0, high1d)
    ax4.set_ylim(ybounds[0], ybounds[1])

    return fig, ax1, ax3, ax4


def create_triangular_fig_and_axes(xbounds, ybounds, figsize=(9.7, 9.7)):
    fig, ax1, ax3, ax4 = triangle_plot_2d_axes(
        xbounds, ybounds, figsize=figsize, width_ratios=[4, 1],
        height_ratios=[1, 4], wspace=0.0, hspace=0.0)
    return fig, ax1, ax3, ax4


def add_samples_to_triangular_fig(
    samples, waveform_model, xlims, ylims, zorder, ax1, ax3, ax4, ptype="full",
    plot_density=True, color=None, linestyle=None, kde=None, kde_kwargs={},
    smooth=7., skip_1d=False, kde_1d=gaussian_kde, kde_1d_kwargs={},
    hpd=False, level=0.9, linewidth=None
):
    x = np.array(samples[0])
    y = np.array(samples[1])
    xlow, xhigh = xlims
    ylow, yhigh = ylims
    if color is None:
        color = color_dict[waveform_model]
    if linestyle is None:
        linestyle = linestyle_dict[waveform_model]
    if linewidth is None:
        linewidth = linewidth_dict[waveform_model]
    if not skip_1d:
        add_samples_to_fig(
            x, xlims, label_dict[waveform_model], alpha_dict[waveform_model],
            linestyle, color, ax1, zorder=zorder, no_fill=not plot_density,
            ptype=ptype, linewidth=linewidth,
            kde_kernel=kde_1d, kde_kwargs=kde_1d_kwargs, hpd=hpd
        )
    if not skip_1d:
        add_samples_to_fig(
            y, ylims, label_dict[waveform_model], alpha_dict[waveform_model],
            linestyle, color, ax4, zorder=zorder, no_fill=not plot_density,
            swap_axis=True, ptype=ptype, linewidth=linewidth,
            kde_kernel=kde_1d, kde_kwargs=kde_1d_kwargs, hpd=hpd
        )
    range = [[xlow, xhigh], [ylow, yhigh]]
    if kde is None:
        corner.hist2d(x, y, bins=300, ax=ax3, range=range, levels=[level], color=color,
                      plot_datapoints=False, plot_density=plot_density, smooth=smooth,
                      fill_contours=False, no_fill_contours=False, zorder=zorder,
                      contour_kwargs=dict(linestyles=[linestyle],
                                          linewidths=[linewidth]))
    else:
        from pesummary.core.plots.corner import hist2d
        hist2d(x, y, bins=300, ax=ax3, range=range, levels=[level], color=color,
                      plot_datapoints=False, plot_density=plot_density, smooth=smooth,
                      fill_contours=False, no_fill_contours=False, zorder=zorder,
                      contour_kwargs=dict(linestyles=[linestyle],
                                          linewidths=[linewidth]),
                      linestyles=[linestyle], kde=kde, kde_kwargs=kde_kwargs)


def combine_samples(waveform_models, outfile):
    data = np.genfromtxt(result_file_map[waveform_models[0]], names=True)
    params = data.dtype.names
    nsamples = len(data)
    dictionary = {
        param: [i[params.index(param)] for i in data] for param in params
    }
    for num, waveform_model in enumerate(waveform_models[1:]):
        data = np.genfromtxt(result_file_map[waveform_model], names=True)
        params = data.dtype.names
        tmp_dict = {
            param: [i[params.index(param)] for i in data] for param in params
        }
        for key in params:
            if key in dictionary.keys():
                dictionary[key] += tmp_dict[key]
    new_dictionary = {}
    for key in dictionary.keys():
        if len(dictionary[key]) != nsamples:
            new_dictionary[key] = dictionary[key]
    
    params = list(new_dictionary.keys())
    samples = np.array([dictionary[param] for param in params])
    np.savetxt(outfile, samples.T, delimiter="\t", header="\t".join(params), comments="")
    return

from scipy.stats import gaussian_kde as kde
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
from matplotlib import patheffects as PathEffects
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.axisartist.grid_finder import MaxNLocator
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper

rc_params = {'backend': 'ps',
             'axes.labelsize': 11,
             'axes.titlesize': 10,
             'font.size': 11,
             'legend.fontsize': 10,
             'xtick.labelsize': 11,
             'ytick.labelsize': 11,
             #'text.usetex': True,
             'font.family': 'Times New Roman'}#,
             #'font.sans-serif': ['Bitstream Vera Sans']}#,

plt.rcParams.update(rc_params)

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

class LVK_Bounded_2d_kde(kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator  for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(LVK_Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(LVK_Bounded_2d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(LVK_Bounded_2d_kde, self).evaluate([2*self.xlow - x, y])

        if self.xhigh is not None:
            pdf += super(LVK_Bounded_2d_kde, self).evaluate([2*self.xhigh - x, y])

        if self.ylow is not None:
            pdf += super(LVK_Bounded_2d_kde, self).evaluate([x, 2*self.ylow - y])

        if self.yhigh is not None:
            pdf += super(LVK_Bounded_2d_kde, self).evaluate([x, 2*self.yhigh - y])

        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(LVK_Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.ylow - y])

            if self.yhigh is not None:
                pdf += super(LVK_Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(LVK_Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(LVK_Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.yhigh - y])

        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results
