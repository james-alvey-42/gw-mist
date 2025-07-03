import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

def housestyle_rcparams(fontsize=15, linewidth=1):
    mpl.rcParams["axes.formatter.use_mathtext"]=True
    mpl.rcParams['font.family']='serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif']=cmfont.get_name()
    mpl.rcParams['mathtext.fontset']='cm'
    mpl.rcParams['axes.unicode_minus']=False
    mpl.rcParams['text.latex.preamble'] = r'\boldmath'
    mpl.rcParams['axes.linewidth'] = linewidth
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['xtick.labelsize']=fontsize
    mpl.rcParams['ytick.labelsize']=fontsize
    mpl.rcParams['font.size']=fontsize
    return True



def cross_axis_locator(figure,axis1,axis2,location): #location is [[xlims],[ylims]]
    x_zoom_min, x_zoom_max = location[0]
    y_zoom_min, y_zoom_max = location[1]
    # ax2.set_xlim(x_zoom_min, x_zoom_max)
    # ax2.set_ylim(y_zoom_min, y_zoom_max)

    # Draw a rectangle around the zoomed area in ax1

    rect = patches.Rectangle(
        (x_zoom_min, y_zoom_min),  # Bottom-left corner
        x_zoom_max - x_zoom_min,  # Width
        y_zoom_max - y_zoom_min,  # Height
        linewidth=1.5,
        edgecolor='red',
        facecolor='none'
    )
    axis1.add_patch(rect)

    # Connect rectangle corners in ax1 to ax2
    rect_corners = [
        (x_zoom_min, y_zoom_min),  # Bottom-left
        (x_zoom_max, y_zoom_min),  # Bottom-right
        (x_zoom_min, y_zoom_max),  # Top-left
        (x_zoom_max, y_zoom_max),  # Top-right
    ]
    lines = [] 
    # Plot lines connecting rectangle corners to ax2
    for corner in rect_corners:
        # Transform corner from data to display coordinates
        ax1_corner_display = axis1.transData.transform(corner)
        ax1_corner_figure = figure.transFigure.inverted().transform(ax1_corner_display)

        # Determine the corresponding side of ax2
        ax2_x = 1 if corner[0] > (x_zoom_min + x_zoom_max) / 2 else 0
        ax2_y = 1 if corner[1] > (y_zoom_min + y_zoom_max) / 2 else 0

        # Transform ax2 edge to figure coordinates
        ax2_edge_display = axis2.transAxes.transform((ax2_x, ax2_y))
        ax2_edge_figure = figure.transFigure.inverted().transform(ax2_edge_display)

        # Compute line length
        line_length = np.sqrt((ax2_edge_figure[0] - ax1_corner_figure[0])**2 +
                            (ax2_edge_figure[1] - ax1_corner_figure[1])**2)
        
        # Add line and its length to the list
        lines.append(((ax1_corner_figure, ax2_edge_figure), line_length))

    # Sort lines by length and remove the two longest
    lines = sorted(lines, key=lambda x: x[1])[:-2]

    # Draw the remaining lines
    for (start, end), _ in lines:
        line = plt.Line2D(
            [start[0], end[0]],
            [start[1], end[1]],
            transform=figure.transFigure,
            color="red",
            linewidth=1.5,
            linestyle='dashed',
            zorder=1
        )
        figure.add_artist(line)
    return True

def create_plot(size = (4,3)):
    fig, ax0 = plt.subplots(figsize = size)
    ax0.set_visible(False)
    ax1 = fig.add_axes((0,0,1,1))
    return fig,ax1

def do_ticks(list_of_axes):
    for ax in list_of_axes:
        ax.minorticks_on()
        ax.tick_params(top=True,right=True, direction='in', length=7, which='major')
        ax.tick_params(top=True,right=True, direction='in', length=4, which='minor')


### FIXING TOOLS FOR GWPY SETTINGS ###
def fix_frame(ax):
    ax.tick_params(color='black', labelcolor='black')
    ax.spines[:].set_color('black')
    ax.spines[:].set_linewidth(1)
    return True

def fix_plot(a):
    for axes in a:
        axes.grid(False)
        do_ticks([axes])
        fix_frame(axes)
    return True