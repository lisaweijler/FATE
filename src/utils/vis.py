import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import math
import abc
from collections import Counter
from pathlib import Path
from typing import Dict, List


from ..utils.loggingmanager import LoggingManager






def mrd_plot(mrd_list_gt, mrd_list_pred, f1_score, filenames=None):
    """
    Creates a plot for true and predicted mrd values.
    Interactive, for wandb
    """

    if filenames is None:
        filenames = [str(n) for n in range(len(mrd_list_gt))]

    min_val = 1.0e-5

    # Set min mrd to min_val:
    mrd_list_gt = [mrd if mrd >= min_val else min_val for mrd in mrd_list_gt]
    mrd_list_pred = [mrd if mrd >= min_val else min_val for mrd in mrd_list_pred]

    # Create figure
    data = pd.DataFrame(list(zip(mrd_list_gt, mrd_list_pred, f1_score, filenames)), 
                        columns=['gt', 'pred', 'f1_score', 'names'])
    fig = px.scatter(data,
                     x='gt', y='pred', 
                     log_x=True, log_y=True, range_x=[min_val, 1], range_y=[min_val, 1],
                     color='f1_score', template='simple_white', hover_name='names')
    
    # Add diagonal line
    fig.add_shape(type='line',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(
            color='Gray',
            width=2,
            dash='dashdot',
        )
    )
    
    # Add vertical line
    fig.add_shape(type='line',
        x0=5.0e-4, y0=0, x1=5.0e-4, y1=1,
        line=dict(
            color='Gray',
            width=2,
            dash='dot',
        )
    )
    
    # Add horizontal line
    fig.add_shape(type='line',
        x0=0, y0=5.0e-4, x1=1, y1=5.0e-4,
        line=dict(
            color='Gray',
            width=2,
            dash='dot',
        )
    )
    return fig



class BasePlot(abc.ABC):

    SAVE_AS_SVG_IN_ADDITION = False

    def __init__(self, enabled: bool, filepath: Path, caption: str, ax: matplotlib.axes = plt.gca()):
        self.enabled = enabled
        self.filepath = filepath
        self.caption = caption
        self.ax = ax
        matplotlib.rcParams['axes.linewidth'] = 2

    def showPlot(self):
        if not self.enabled:
            return
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def savePlot(self):
        if not self.enabled:
            return
        plt.savefig(self.filepath, dpi=80)
        if BasePlot.SAVE_AS_SVG_IN_ADDITION and self.filepath.name.endswith(".png"):
            plt.savefig(self.filepath.parent / Path(self.filepath.name.replace(".png", ".svg")), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()
        LoggingManager.get_default_logger().info(f"successfully saved plot with caption '{self.caption}' to path: '{self.filepath}'")

    @abc.abstractmethod
    def createPlot(self):
        pass

    def generatePlotFile(self):
        if not self.enabled:
            return
        self.createPlot()
        self.savePlot()


class PanelPlot(BasePlot):

    def __init__(self, enabled: bool, filepath: Path, caption: str, 
                       events: pd.DataFrame, panels: List[List[str]], 
                       num_plots_per_row: int, min_fig_size: int,
                       color: pd.Series, palette, hue_order, 
                       point_types=None, ax: matplotlib.axes = plt.gca(), use_new_ax=True):

        self.data = events.copy()
        self.panels = panels
        self.num_plots_per_row = num_plots_per_row
        self.marker_list = self.data.columns
        self.min_fig_size = min_fig_size
        self.data["color"] = color.reset_index(drop=True)
        self.data["size"] = 0.2
        self.palette = palette
        self.hue_order = hue_order
        self.use_new_ax = use_new_ax
        self.point_types = point_types
        self.eventsInfoString = "".join(["{}:{} events, ".format(*i) for i in Counter(color).items()])
    
        super(PanelPlot, self).__init__(enabled, filepath, caption, ax)

    def createPlot(self):
        """
        Creates a scatter plot for the marker combinations given in theconfig panel.
        """
        num_plots = len(self.panels)

        num_rows = math.ceil(num_plots/self.num_plots_per_row)

        fig, ax = plt.subplots(num_rows, self.num_plots_per_row, figsize=(
            self.min_fig_size*self.num_plots_per_row, self.min_fig_size*num_rows))
        plt.suptitle(self.caption + "\n" + self.eventsInfoString)


        for r in range(num_rows):
            for p in range(self.num_plots_per_row):
                panel_idx = (r*self.num_plots_per_row) + p
                if panel_idx > num_plots-1:
                    break
                df_to_plot = self.data[self.panels[panel_idx] + ['color', 'size']]




                scat_plt = sns.scatterplot(data=df_to_plot, 
                                x=self.panels[panel_idx][0], y=self.panels[panel_idx][1],
                                hue="color", palette=self.palette, hue_order=self.hue_order, 
                                ax=ax[r,p], alpha=0.9, size='size', sizes=(0.2, 0.2))

                scat_plt.set_ylim(-0.3, 4.6)
                scat_plt.set_xlim(-0.3, 4.6)


class PanelPlotTargetVSPrediction(BasePlot):
    '''
    TODO: update for multiclass prediction, now only for binary classification
    '''

    def __init__(self, enabled: bool,
                       filepath: Path, 
                       caption: str, 
                       events: np.ndarray,
                       target: np.ndarray, 
                       prediction: np.ndarray,
                       marker_list: List[str],
                       panels: List[List[str]], 
                       min_fig_size: int,
                       n_points: int = 10000,
                       ax: matplotlib.axes = plt.gca(), 
                       use_new_ax=True):

        self.data = events
        self.target = target
        self.prediction = prediction
        self.panels = panels
        self.marker_list = marker_list
        self.min_fig_size = min_fig_size
        self.use_new_ax = use_new_ax
        self.n_points = n_points
    
        super(PanelPlotTargetVSPrediction, self).__init__(enabled, filepath, caption, ax)

    def createPlot(self):
        """
        Creates a scatter plot for the marker combinations given in theconfig panel.
        """
        num_plots = len(self.panels)

        

        fig, ax = plt.subplots(2, num_plots, figsize=(
            self.min_fig_size*num_plots, self.min_fig_size*2))

        plt.suptitle(self.caption)

        for p in range(num_plots):
            m0_str = self.panels[p][0]
            m0 = self.marker_list.index(m0_str)
            m1_str = self.panels[p][1]
            m1 = self.marker_list.index(m1_str)

            x_data = self.data[:self.n_points, m0]
            y_data = self.data[:self.n_points, m1]
            colors_GT = ['red' if x > 0.5 else 'blue' for x in self.target[:self.n_points]]
            colors_pred = ['red' if x > 0.5 else 'blue' for x in self.prediction[:self.n_points]]

            marker_size_GT = [1.5 if colors_GT[n] ==
                            'red' else 0.75 for n, m in enumerate(colors_GT)]
            marker_size_pred = [1.5 if colors_GT[n] ==
                            'red' else 0.75 for n, m in enumerate(colors_pred)]


            ax[0, p].scatter(x_data, y_data, s=marker_size_pred, c=colors_pred)
            ax[0, p].set_xlabel(m0_str)
            ax[0, p].set_ylabel(m1_str)                
            

            ax[1, p].scatter(x_data, y_data, s=marker_size_GT, c=colors_GT)
            ax[1, p].set_xlabel(m0_str)
            ax[1, p].set_ylabel(m1_str)

            if p == 0:
                ax[0, p].title.set_text('Prediction')
                ax[1, p].title.set_text('Target')

