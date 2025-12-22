#Script to reproduce Fig 3, 4, 7, 9

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from pesummary.gw.file.read import read
from pesummary.gw.fetch import fetch_open_samples
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.utils.bounded_2d_kde import Bounded_2d_kde
from pesummary.gw.plots.publication import _return_bounds
from pesummary.gw.plots.cmap import colormap_with_fixed_hue
os.environ["GWPY_RCPARAMS"] = "False"

from utils import *
import corner

plt.style.use("MATPLOTLIB_RCPARAMS.sty")
xlims = [40., 180.]
ylims = [30., 140.]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["mass_1_source", "mass_2_source"]
ax1.axvspan(60, 130, color='tab:orange', alpha=0.12)
ax4.axhspan(60, 130, color='tab:orange', alpha=0.12)

interested=["combined", "NRSur7dq4"]
for waveform_model in interested:
    samples = samples_map[waveform_model]
    if waveform_model == "NRSur7dq4":
        linewidth = 2.0
    else:
        linewidth = None
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [samples[PLOTTING_PARAMETERS[0]], samples[PLOTTING_PARAMETERS[1]]], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=1.0,
        linewidth=linewidth, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )

#Open posterior from GW190521 (Fig 3)
gw190521 = fetch_open_samples("GW190521", outdir="./", catalog="GWTC-2", unpack=True, path="GW190521/GW190521.h5")
gw190521_samples = gw190521.samples_dict["PublicationSamples"]
add_samples_to_triangular_fig(
    [gw190521_samples[PLOTTING_PARAMETERS[0]], gw190521_samples[PLOTTING_PARAMETERS[1]]], "combined",
    xlims, ylims, -10, ax1, ax3, ax4, ptype="full", plot_density=False, kde=Bounded_2d_kde, smooth=1.0,
    color="tab:red", linestyle="-", linewidth=2.0, skip_1d=True, kde_kwargs={
        "transform": transform, "xlow": xlow, "xhigh": xhigh,
        "ylow": ylow, "yhigh": yhigh
    }
)

#Load posterior predictive distribution (PPD) for the largest BH mass in mock catalogs similar to GWTC-3 (Fig 3)
with open("../../Results/Fig3_mmax_PPD.json", "r") as f:
    ppd = json.load(f)['mass_1_source']['observable']
add_samples_to_fig(ppd, xlims, r"Highest mass PPD", 0.2, "--", "tab:blue", ax1, swap_axis=False, ptype=None)
add_samples_to_fig(ppd, xlims, r"Highest mass PPD", 0.2, "--", "tab:blue", ax4, swap_axis=True, ptype=None)

ax1.set_ylim(0, 0.05)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 0.05)
ax3.set_xlabel(GWlatex_labels["mass_1"])
ax3.set_ylabel(GWlatex_labels["mass_2"])

symaxis = plt.Polygon([[0, 0], [0, 1000], [1000, 1000]], color='k', alpha=0.05)
ax3.add_patch(symaxis)

handles = [
    mlines.Line2D([], [], color=color_dict["combined"], linestyle=linestyle_dict["combined"], label=label_dict["combined"]),
    mlines.Line2D([], [], color=color_dict["NRSur7dq4"], linestyle=linestyle_dict["NRSur7dq4"], label=label_dict["NRSur7dq4"]),
    mlines.Line2D([], [], color="tab:red", linestyle="-", label= "GW190521"),
    mlines.Line2D([], [], color="tab:blue", linestyle="--", label=r"$m_\mathrm{max}^\mathrm{obs}$ PPD"),
]
ax3.legend(handles=handles, handler_map=None, handlelength=1.5, loc="upper left", frameon=True, fancybox=True, fontsize=25, facecolor='w')
plt.tight_layout()
for ax in [ax1, ax3, ax4]:
    ax.grid(visible=False)
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.grid(visible=False)
fig.savefig("../../Figures/mass_1_mass_2_contour.pdf", format="pdf")

xlims = [0.38, 1.005]
ylims = [0., 1.005]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["a_1", "a_2"]

interested=["combined", "NRSur7dq4"]
for waveform_model in interested:
    samples = samples_map[waveform_model]
    add_samples_to_triangular_fig(
        [samples[PLOTTING_PARAMETERS[0]], samples[PLOTTING_PARAMETERS[1]]], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=5.0,
        level=0.9, kde_1d=bounded_1d_kde, kde_kwargs={
            "xlow": 0, "xhigh": 1, "ylow": 0, "yhigh": 1
        },
        kde_1d_kwargs={"xlow": 0, "xhigh": 1}, hpd=True
    )

ax1.set_ylim(0, 7.0)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 6.5)
ax3.set_xlabel(r"$\chi_{1}$")
ax3.set_ylabel(r"$\chi_{2}$")
ax3.legend(*ax4.get_legend_handles_labels(), loc="upper left", frameon=False, fontsize=25, handler_map=None, handlelength=1.5)
plt.tight_layout()
for ax in [ax1, ax3, ax4]:
    ax.grid(visible=False)
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
fig.savefig("../../Figures/a_1_a_2_contour.pdf", format="pdf")

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
plt.rcParams.update(rc_params)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
H1 = read("../../Results/posterior_samples_H1.h5")
L1 = read("../../Results/posterior_samples_L1.h5")
H1_samples = H1.samples_dict["C00:Mixed"]
L1_samples = L1.samples_dict["C00:Mixed"]
H1L1_samples = samples_map["combined"]
custom_cmap = colormap_with_fixed_hue(color_dict["combined"])
for selected_run, ifo, cmap in zip([H1_samples, L1_samples, H1L1_samples], ["H1", "L1", "H1L1"], [plt.cm.Blues, plt.cm.Greens, custom_cmap]):

    scale = 0.99/0.99

    spin_costilt_1 = []
    spin_costilt_2 = []

    for j in np.arange(0,len(selected_run['a_1'])):
        spin_costilt_1.append([selected_run['a_1'][j],np.cos(selected_run['tilt_1'][j])])
        spin_costilt_2.append([selected_run['a_2'][j],np.cos(selected_run['tilt_2'][j])])

    spin_costilt_1 = np.array(spin_costilt_1)
    spin_costilt_2 = np.array(spin_costilt_2)
    spin1 = LVK_Bounded_2d_kde(spin_costilt_1.reshape(-1, 2), xlow=0, xhigh=.99*scale, ylow=-1, yhigh=1)
    spin2 = LVK_Bounded_2d_kde(spin_costilt_2.reshape(-1, 2), xlow=0, xhigh=.99*scale, ylow=-1, yhigh=1)

    Na, Nt = 30, 30

    # Coordmatplotlib KDE
    rs = np.linspace(0, .99*scale, Na)
    dr = np.abs(rs[1] - rs[0])

    costs = np.linspace(-1, 1, Nt)
    dcost = np.abs(costs[1] - costs[0])

    COSTS, RS = np.meshgrid(costs[:-1], rs[:-1])

    # Coords for plotting
    X = np.arccos(COSTS) * 180/np.pi + 90.
    Y = RS

    H1 = spin1(np.column_stack([RS.ravel()+dr, COSTS.ravel()+dcost]))
    H2 = spin2(np.column_stack([RS.ravel()+dr, COSTS.ravel()+dcost]))

    H1 = H1/np.sum(H1)*1000
    H2 = H2/np.sum(H2)*1000

    #vmax = max(H1.max(), H2.max())
    vmax = 8

    fig = plt.figure(figsize=(5.2, 6))

    # Spin 1
    rect = 121

    tr = Affine2D().translate(90, 0) + Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorD(7)
    tick_formatter1 = angle_helper.FormatterDMS()

    grid_locator2 = MaxNLocator(5)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, 180, 0, .99*scale),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=None)

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # Label angles on the outside
    ax1.axis["bottom"].toggle(all=False)
    ax1.axis["top"].toggle(all=True)
    ax1.axis["top"].major_ticks.set_tick_out(True)

    # Labels on the outside
    ax1.axis["top"].set_axis_direction("top")
    ax1.axis["top"].set_ticklabel_direction('+')

    # Label the radii
    ax1.axis["left"].major_ticks.set_tick_out(True)
    ax1.axis["left"].set_axis_direction('right')

    patches = []
    colors = []
    for x, y, h in zip(X.ravel(), Y.ravel(), H1.ravel()):
        cosx = np.cos((x - 90)*np.pi/180)
        cosxp = cosx + dcost
        xp = np.arccos(cosxp)
        xp = xp*180./np.pi + 90.
        patches.append(Wedge((0., 0.), y+dr, xp, x, width=dr))
        colors.append(h)

    p = PatchCollection(patches, cmap=cmap, edgecolors='face')
    p.set_clim(0, vmax)
    p.set_array(np.array(colors))
    ax1.add_collection(p)

    # Spin 2

    rect = 122

    tr_rotate = Affine2D().translate(90, 0)
    tr_scale = Affine2D().scale(np.pi/180., 1.)
    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorD(7)
    tick_formatter1 = angle_helper.FormatterDMS()

    grid_locator2 = MaxNLocator(5)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, 180, 0, .99*scale),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=None)

    ax2 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    ax2.invert_xaxis()
    fig.add_subplot(ax2)

    # Label angles on the outside
    ax2.axis["bottom"].toggle(all=False)
    ax2.axis["top"].toggle(all=True)
    ax2.axis["top"].set_axis_direction("top")
    ax2.axis["top"].major_ticks.set_tick_out(True)

    # Remove radial labels
    ax2.axis["left"].major_ticks.set_tick_out(True)
    ax2.axis["right"].major_ticks.set_tick_out(True)
    ax2.axis["left"].toggle(ticklabels=False)

    patches = []
    colors = []
    for x, y, h in zip(X.ravel(), Y.ravel(), H2.ravel()):
        cosx = np.cos((x - 90)*np.pi/180)
        cosxp = cosx + dcost
        xp = np.arccos(cosxp)
        xp = xp*180./np.pi + 90.
        patches.append(Wedge((0., 0.), y+dr, xp, x, width=dr))
        colors.append(h)

    p = PatchCollection(patches, cmap=cmap, edgecolors='face')
    p.set_clim(0, vmax)
    p.set_array(np.array(colors))
    ax2.add_collection(p)

    fig.subplots_adjust(wspace=0.18)


    plt.text(1.1*scale, +1.15*scale, r'$c\mathbf{S}_{1}/(Gm_1^2)$', fontsize=14)
    plt.text(-.35*scale, +1.15*scale, r'$c\mathbf{S}_{2}/(Gm_2^2)$', fontsize=14)

    # Annotate axes
    plt.text(-.8*scale, .9*scale, r'$\mathrm{tilt}$', fontsize=14)
    txt = plt.text(-.09*scale, 0.05*scale, r'$\mathrm{magnitude}$', fontsize=14)
    txt.set_path_effects([PathEffects.Stroke(linewidth=1.5, foreground="w"), PathEffects.Normal()])



    aux_ax2 = ax2.get_aux_axes(tr)

    plt.text(-.9*scale, -1.407*scale, r'$\times 10^{-3}$', fontsize=11)

    axcb = fig.colorbar(p,fig.add_axes([0.18, 0.1, 0.66, 0.03]),orientation='horizontal', extend="max")
    axcb.set_label('posterior probability per pixel',fontsize=14)

    # gridline
    ax1.grid(True)
    ax2.grid(True)

    gridlines = ax1.get_xgridlines() + ax1.get_ygridlines() + \
                ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines:
        line.set_alpha(0.5)



    # Tilt label
    txt = aux_ax2.annotate("",
                           xy=(55, 1.158*scale), xycoords='data',
                           xytext=(35, 1.158*scale), textcoords='data',
                           arrowprops=dict(arrowstyle="->",
                                           color="k",
                                           shrinkA=2, shrinkB=2,
                                           patchA=None,
                                           patchB=None,
                                           connectionstyle='arc3,rad=-0.16'))
    txt.arrow_patch.set_path_effects([
        PathEffects.Stroke(linewidth=2, foreground="w"),
        PathEffects.Normal()])

    # Magnitude label
    txt = aux_ax2.annotate("",
                           xy=(88, .25*scale), xycoords='data',
                           xytext=(30, .0*scale), textcoords='data',
                           arrowprops=dict(arrowstyle="->",
                                           color="k",
                                           shrinkA=2, shrinkB=2,
                                           patchA=None,
                                           patchB=None))


    txt.arrow_patch.set_path_effects([
        PathEffects.Stroke(linewidth=2, foreground="w"),
        PathEffects.Normal()])


    aux_ax1 = ax1.get_aux_axes(tr)


    txt.arrow_patch.set_path_effects([
        PathEffects.Stroke(linewidth=2, foreground="w"),
        PathEffects.Normal()])

    fig.savefig(f"../../Figures/spin_disk_plot_GW231123_{ifo}.pdf")
