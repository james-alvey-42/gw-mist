#Script to reproduce Fig 8

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
import json
import os
import bilby
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from pesummary.io import read
from pesummary.gw.plots.latex_labels import GWlatex_labels
from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.utils.bounded_2d_kde import Bounded_2d_kde
from pesummary.gw.plots.publication import _return_bounds
from pesummary.gw.plots.cmap import colormap_with_fixed_hue
from pesummary.core.plots.corner import corner
from utils import *
plt.style.use("MATPLOTLIB_RCPARAMS.sty")
os.environ["GWPY_RCPARAMS"] = "False"

xlims = [50., 180.]
ylims = [30., 140.]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["mass_1_source", "mass_2_source"]

handles = []
interested=["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]
for waveform_model in interested:
    samples = samples_map[waveform_model]
    xx = samples[PLOTTING_PARAMETERS[0]]
    yy = samples[PLOTTING_PARAMETERS[1]]
    if waveform_model in ["IMRPhenomXPHM", "IMRPhenomXO4a"]:
        ls = "--"
    elif waveform_model in ["IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]:
        ls = "-"
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [xx, yy], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=2.0,
        linestyle=ls, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )
    handles.append(mlines.Line2D([], [], color=color_dict[waveform_model], linestyle=ls, label=label_dict[waveform_model]))

symaxis = plt.Polygon([[0, 0], [0, 1000], [1000, 1000]], color='k', alpha=0.05)
ax3.add_patch(symaxis)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax1.set_ylim(0, 0.08)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 0.08)
ax3.set_xlabel(r"$m_{1} [M_{\odot}$]")
ax3.set_ylabel(r"$m_{2} [M_{\odot}$]")
leg = ax3.legend(handles=handles, handler_map=None, handlelength=2, loc="upper left", frameon=True, fancybox=True, fontsize=25, facecolor='w')
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.savefig("../../Figures/mass_1_mass_2_contour_individual_models.pdf")
plt.close()

xlims = [240., 420.]
ylims = [0.1, 1.]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["total_mass", "mass_ratio"]
handles = []
interested=["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]
for waveform_model in interested:
    samples = samples_map[waveform_model]
    xx = samples[PLOTTING_PARAMETERS[0]]
    yy = samples[PLOTTING_PARAMETERS[1]]
    if waveform_model in ["IMRPhenomXPHM", "IMRPhenomXO4a"]:
        ls = "--"
    elif waveform_model in ["IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]:
        ls = "-"
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [xx, yy], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=2.0,
        linestyle=ls, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )
    handles.append(mlines.Line2D([], [], color=color_dict[waveform_model], linestyle=ls, label=label_dict[waveform_model]))

ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax1.set_ylim(0, 0.05)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 11.0)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax3.set_xlabel(r"$(1 + z) M [M_{\odot}$]")
ax3.set_ylabel(r"$q$")
leg = ax3.legend(handles=handles, handler_map=None, handlelength=2, loc="lower right", frameon=False, fancybox=True, fontsize=25, facecolor='w')
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.savefig("../../Figures/total_mass_mass_ratio_contour_individual_models.pdf", bbox_extra_artists=[leg])
plt.close()

xlims = [90, 195]
ylims = [40, 120]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["mass_1_source", "mass_2_source"]

result_0483 = read("../../Results/SXS_BBH_0483_posterior_samples.h5")
samples_0483 = result_0483.samples_dict
nrsur_0483_samples = samples_0483["C00:NRSur7dq4"]
xphm_0483_samples = samples_0483["C00:IMRPhenomXPHM-SpinTaylor"]
tphm_0483_samples = samples_0483["C00:IMRPhenomTPHM"]
xo4a_0483_samples = samples_0483["C00:IMRPhenomXO4a"]
seob_0483_samples = samples_0483["C00:SEOBNRv5PHM"]

result_4030 = read("../../Results/SXS_BBH_4030_posterior_samples.h5")
samples_4030 = result_4030.samples_dict
nrsur_4030_samples = samples_4030["C00:NRSur7dq4"]
xphm_4030_samples = samples_4030["C00:IMRPhenomXPHM-SpinTaylor"]
tphm_4030_samples = samples_4030["C00:IMRPhenomTPHM"]
xo4a_4030_samples = samples_4030["C00:IMRPhenomXO4a"]
seob_4030_samples = samples_4030["C00:SEOBNRv5PHM"]

interested=["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]
handles = []
for num, waveform_model in enumerate(interested):
    if waveform_model == "IMRPhenomXPHM":
        samples = xphm_0483_samples
    elif waveform_model == "IMRPhenomXO4a":
        samples = xo4a_0483_samples
    elif waveform_model == "IMRPhenomTPHM":
        samples = tphm_0483_samples
    elif waveform_model == "NRSur7dq4":
        samples = nrsur_0483_samples
    elif waveform_model == "SEOBNRv5PHM":
        samples = seob_0483_samples
    if waveform_model in ["IMRPhenomXPHM", "IMRPhenomXO4a"]:
        ls = "--"
    elif waveform_model in ["IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]:
        ls = "-"
    xx = samples[PLOTTING_PARAMETERS[0]]
    yy = samples[PLOTTING_PARAMETERS[1]]
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [xx, yy], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=2.0,
        linestyle=ls, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )
    handles.append(mlines.Line2D([], [], color=color_dict[waveform_model], linestyle=ls, label=label_dict[waveform_model]))
   
ax3.axvline(result_0483.injection_parameters[0][PLOTTING_PARAMETERS[0]][0], color='k', linewidth=2)
ax3.axhline(result_0483.injection_parameters[0][PLOTTING_PARAMETERS[1]][0], color='k', linewidth=2)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax1.set_ylim(0, 0.05)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 0.06)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax3.legend(handles=handles, handler_map=None, handlelength=2, loc="lower right", frameon=False, fancybox=True, fontsize=25, facecolor='w')
ax3.set_xlabel(r"$m_{1} [M_{\odot}$]")
ax3.set_ylabel(r"$m_{2} [M_{\odot}$]")
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.savefig("../../Figures/mass_1_mass_2_contour_individual_models_0483.pdf") #, bbox_extra_artists=[leg])
plt.close()

xlims = [75, 280]
ylims = [10, 60]

fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["mass_1_source", "mass_2_source"]

interested=["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]
handles = []
for num, waveform_model in enumerate(interested):
    if waveform_model == "IMRPhenomXPHM":
        samples = xphm_4030_samples
    elif waveform_model == "IMRPhenomXO4a":
        samples = xo4a_4030_samples
    elif waveform_model == "IMRPhenomTPHM":
        samples = tphm_4030_samples
    elif waveform_model == "NRSur7dq4":
        samples = nrsur_4030_samples
    elif waveform_model == "SEOBNRv5PHM":
        samples = seob_4030_samples
    if waveform_model in ["IMRPhenomXPHM", "IMRPhenomXO4a"]:
        ls = "--"
    elif waveform_model in ["IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]:
        ls = "-"
    xx = samples[PLOTTING_PARAMETERS[0]]
    yy = samples[PLOTTING_PARAMETERS[1]]
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [xx, yy], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=2.0,
        linestyle=ls, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )
    handles.append(mlines.Line2D([], [], color=color_dict[waveform_model], linestyle=ls, label=label_dict[waveform_model]))

ax3.axvline(result_4030.injection_parameters[0][PLOTTING_PARAMETERS[0]][0], color='k', linewidth=2)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax1.set_ylim(0, 0.07)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 0.15)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax3.legend(handles=handles, handler_map=None, handlelength=2, loc="upper right", frameon=False, fancybox=True, fontsize=25, facecolor='w')
ax3.set_xlabel(r"$m_{1} [M_{\odot}$]")
ax3.set_ylabel(r"$m_{2} [M_{\odot}$]")
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.savefig("../../Figures/mass_1_mass_2_contour_individual_models_4030.pdf") #, bbox_extra_artists=[leg])
plt.close()

xlims = [200., 350.]
ylims = [0.3, 1.0]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["total_mass", "mass_ratio"]

interested=["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]
handles = []
for num, waveform_model in enumerate(interested):
    if waveform_model == "IMRPhenomXPHM":
        samples = xphm_0483_samples
    elif waveform_model == "IMRPhenomXO4a":
        samples = xo4a_0483_samples
    elif waveform_model == "IMRPhenomTPHM":
        samples = tphm_0483_samples
    elif waveform_model == "NRSur7dq4":
        samples = nrsur_0483_samples
    elif waveform_model == "SEOBNRv5PHM":
        samples = seob_0483_samples
    if waveform_model in ["IMRPhenomXPHM", "IMRPhenomXO4a"]:
        ls = "--"
    elif waveform_model in ["IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]:
        ls = "-"
    xx = samples[PLOTTING_PARAMETERS[0]]
    yy = samples[PLOTTING_PARAMETERS[1]]
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [xx, yy], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=2.0,
        linestyle=ls, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )
    handles.append(mlines.Line2D([], [], color=color_dict[waveform_model], linestyle=ls, label=label_dict[waveform_model]))

ax3.axvline(result_0483.injection_parameters[0][PLOTTING_PARAMETERS[0]][0], color='k', linewidth=2)
ax3.axhline(result_0483.injection_parameters[0][PLOTTING_PARAMETERS[1]][0], color='k', linewidth=2)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax1.set_ylim(0, 0.05)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 8.0)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax3.legend(handles=handles, handler_map=None, handlelength=2, loc="lower left", frameon=False, fancybox=True, fontsize=25, facecolor='w')
ax3.set_xlabel(r"$(1 + z) M [M_{\odot}$]")
ax3.set_ylabel(r"$q$")
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.savefig("../../Figures/total_mass_mass_ratio_contour_individual_models_0483.pdf") #, bbox_extra_artists=[leg])
plt.close()

xlims = [110., 320.]
ylims = [0.05, 0.5]
fig, ax1, ax3, ax4 = create_triangular_fig_and_axes(xlims, ylims)
PLOTTING_PARAMETERS = ["total_mass", "mass_ratio"]

interested=["IMRPhenomXPHM", "IMRPhenomXO4a", "IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]
handles = []
for num, waveform_model in enumerate(interested):
    if waveform_model == "IMRPhenomXPHM":
        samples = xphm_4030_samples
    elif waveform_model == "IMRPhenomXO4a":
        samples = xo4a_4030_samples
    elif waveform_model == "IMRPhenomTPHM":
        samples = tphm_4030_samples
    elif waveform_model == "NRSur7dq4":
        samples = nrsur_4030_samples
    elif waveform_model == "SEOBNRv5PHM":
        samples = seob_4030_samples
    if waveform_model in ["IMRPhenomXPHM", "IMRPhenomXO4a"]:
        ls = "--"
    elif waveform_model in ["IMRPhenomTPHM", "NRSur7dq4", "SEOBNRv5PHM"]:
        ls = "-"
    xx = samples[PLOTTING_PARAMETERS[0]]
    yy = samples[PLOTTING_PARAMETERS[1]]
    transform, xlow, xhigh, ylow, yhigh = _return_bounds(PLOTTING_PARAMETERS)
    add_samples_to_triangular_fig(
        [xx, yy], waveform_model,
        xlims, ylims, -10, ax1, ax3, ax4, ptype="full", kde=Bounded_2d_kde, smooth=2.0,
        linestyle=ls, kde_kwargs={
            "transform": transform, "xlow": xlow, "xhigh": xhigh,
            "ylow": ylow, "yhigh": yhigh
        }
    )
    handles.append(mlines.Line2D([], [], color=color_dict[waveform_model], linestyle=ls, label=label_dict[waveform_model]))

ax3.axvline(result_4030.injection_parameters[0][PLOTTING_PARAMETERS[0]][0], color='k', linewidth=2)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax1.set_ylim(0, 0.05)
ax1.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_xlim(0, 15.0)
ax1.grid(visible=False)
ax3.grid(visible=False)
ax4.grid(visible=False)
ax3.legend(handles=handles, handler_map=None, handlelength=2, loc="upper left", frameon=False, fancybox=True, fontsize=25, facecolor='w')
ax3.set_xlabel(r"$(1 + z) M [M_{\odot}$]")
ax3.set_ylabel(r"$q$")
for ax in [ax1, ax3, ax4]:
    for artist in ax.get_children():
        try:
            artist.set_rasterized(True)
        except AttributeError:
            pass
plt.savefig("../../Figures/total_mass_mass_ratio_contour_individual_models_4030.pdf") #, bbox_extra_artists=[leg])
plt.close()
