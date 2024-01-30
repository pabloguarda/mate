import os
import matplotlib.pyplot as plt
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import collections.abc
from sklearn.linear_model import LinearRegression
plt.style.use('default')

import seaborn as sns
import scipy as sp
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
# from pysal.viz.mapclassify import Natural_Breaks
from shapely.geometry import Point, LineString
sns.set_style('ticks')
sns.set_context('notebook')
# from isuelogit.estimation import compute_vot
# from typing import Union, Dict, List, Tuple
# from isuelogit.mytypes import Matrix
from pesuelogit.models import compute_rr
from mate.models import bpr_function, utility_parameters_periods, compute_baseline_predictions, \
    compute_relative_gap_by_period
from mate.etl import get_tensors_by_year
from mate.metrics import r2_score, mse
import time
from typing import Union, Dict, List, Tuple
from isuelogit.mytypes import Matrix

def plot_hyperparameter_grid_results(df):
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 10))
    fig = plt.figure(figsize=(8, 8))  # Adjust the size as needed
    ax = fig.add_subplot(111, projection='3d')
    df = df.sort_values('lambda_equilibrium')
    original_lambda_equilibrium = df['lambda_equilibrium'].unique()
    original_lambda_equilibrium = np.array([f'{x:.6f}'.rstrip('0').rstrip('.') if '.' in f'{x:.6f}' else str(int(x))
                                            for x in original_lambda_equilibrium])
    # Add a very small constant to show case when lambda is equal to zero
    df['lambda_equilibrium'][df['lambda_equilibrium'] == 0] += 1e-9
    x = np.log10(df[(df.component == 'traveltime')]['value'].values)
    y = np.log10(df[(df.component == 'flow')]['value'].values)
    z = np.log10(df['lambda_equilibrium'].unique())
    c = df[['lambda_equilibrium', 'relative_gap']].sort_values(['lambda_equilibrium'])[
        'relative_gap'].drop_duplicates().values
    p = ax.scatter(x, y, z,
                   c=c,
                   norm=colors.LogNorm(vmin=1e-2, vmax=5),
                   s=40,
                   # cmap='RdBu_r',
                   # cmap='PRGn_r'
                   cmap='Greens_r'
                   )
    cbar = plt.colorbar(p,
                        # ticks=[1e-3,1e-4,1e-5,1e-6,1e-7],
                        # ticks=np.linspace(start = 1e-6, stop = 1e-7,num = 5),
                        cax=fig.add_axes([0.92, 0.3, 0.03, 0.38]))
    cbar.set_label(f'relative gap', rotation=270, labelpad=15, fontsize=18)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    # for i, txt in enumerate(z):
    #     ax.annotate(txt, (x[i], y[i], z[i]))
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], f'{original_lambda_equilibrium[i]}', size=14, zorder=1, color='k')
    ax.set_xlabel(r'$\log(\ell_t)$', fontsize=18, labelpad=10)
    ax.set_ylabel(r'$\log(\ell_x)$', fontsize=18, labelpad=10)
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel(r'$\log(\lambda_{e})$', labelpad=5, fontsize=18, rotation=90)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_box_aspect(aspect=(1, 1, 1.05))

    # x, y, z = 0.5, 0.5, -0.1
    # ax.text(x, y, z, 'Z-axis Label', transform=ax.get_proj(), ha='right', va='center', rotation=90, fontsize=12, color='blue')

    # ax.xaxis.pane.fill = False  # X-axis
    # ax.yaxis.pane.fill = False  # Y-axis
    # ax.zaxis.pane.fill = False  # Z-axis
    ax.view_init(elev=10., azim=-20, roll=0)
    # ax.view_init(elev=10., azim=-25, roll=0)
    # plt.tight_layout()
    return fig, ax

def plot_metrics_kfold(df, metric_name='mape', benchmark_name='historical mean', model_name='MaTE',
                       sharex=True, sharey=True, **kwargs):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(9, 7), sharex=sharex, sharey=sharey)
    df = df.copy()
    df.loc[df.dataset == 'training', 'dataset'] = 'in-sample'
    df.loc[df.dataset == 'validation', 'dataset'] = 'out-of-sample'
    df.loc[df.stage == 'final', 'stage'] = model_name
    df.loc[df.stage == 'final', 'stage'] = model_name
    ax = axs[0, 0]
    sns.boxplot(data=df[(df.stage == 'initial') & (df.metric == metric_name)], x="component", y="value", hue='dataset',
                ax=ax, legend=False, **kwargs)
    # ax.set_title(metric_name + ' before model training')
    ax.set_title('start of model training', fontsize=18)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')
    # ax.legend(loc='upper right')
    ax = axs[0, 1]
    sns.boxplot(data=df[(df.stage == model_name) & (df.metric == metric_name)], x="component", y="value", hue='dataset',
                ax=ax, legend=True, **kwargs)
    # ax.set_title(metric_name + ' after model training')
    ax.set_title('end of model training', fontsize=18)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')
    legend = ax.legend(loc='upper right', title='folds', bbox_to_anchor=(1.7, 0.6), title_fontsize=12, fontsize=12)
    # legend._legend_box.align = "left"
    df = df[df['stage'].isin([benchmark_name, model_name])]
    df['stage'] = pd.Categorical(df['stage'], categories=[benchmark_name, model_name])
    ax = axs[1, 0]
    sns.boxplot(data=df[(df.dataset == 'in-sample')
                        & (df.metric == metric_name) & (df.stage != 'initial')], x="component", y="value", hue='stage',
                ax=ax, legend=False, palette=list(sns.color_palette("deep"))[3:5], **kwargs)
    # ax.set_title(metric_name + ' in training set')
    ax.set_title('in-sample folds', fontsize=18)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')
    # ax.legend(title=None, loc='upper right')
    ax = axs[1, 1]
    sns.boxplot(data=df[(df.dataset == 'out-of-sample')
                        & (df.metric == metric_name) & (df.stage != 'initial')], x="component", y="value", hue='stage',
                ax=ax, legend=True, palette=list(sns.color_palette("deep"))[3:5], **kwargs)
    # ax.axhline(df[(df.dataset == 'validation') & (df.metric == metric_name) & (df.stage == benchmark_name) ])
    # ax.set_title(metric_name + ' in validation set')
    ax.set_title('out-of-sample folds', fontsize=18)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('loss component')
    legend = ax.legend(loc='upper right', title='model', bbox_to_anchor=(1.74, 0.6), title_fontsize=12, fontsize=12)
    # ax.legend(title=None, loc='upper right')
    # legend._legend_box.align = "left"
    for ax in axs.reshape(-1):
        # ax.get_legend().remove()
        # ax.set_xlim(xmin=-100)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.get_yaxis().get_label().set_visible(True)
        ax.get_xaxis().get_label().set_visible(True)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # ax.grid(False)
    axs[0, 1].get_yaxis().get_label().set_visible(False)
    axs[1, 1].get_yaxis().get_label().set_visible(False)
    axs[0, 0].get_xaxis().get_label().set_visible(False)
    axs[0, 1].get_xaxis().get_label().set_visible(False)
    sns.set_style("ticks")
    return fig, axs

def plot_baselines_kfold(df, metric_name='mape', sharex=True, sharey=True, **kwargs):
    # fig, axs = plt.subplots()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(7.5, 5), sharex=sharex, sharey=sharey)
    df = df.copy()
    # fig, ax = plt.subplots()
    sns.boxplot(data=df, x="component", y="value", hue='model', ax=ax, **kwargs)
    # ax.set_title(metric_name + ' before model training')
    # ax.set_title('start of model training')
    ax.set_ylabel(metric_name, fontsize=18)
    ax.set_xlabel('loss component', fontsize=18)
    ax.get_yaxis().get_label().set_visible(True)
    ax.get_xaxis().get_label().set_visible(True)
    ax.xaxis.set_tick_params(which='both', labelbottom=True, labelsize=16)
    ax.yaxis.set_tick_params(which='both', labelleft=True, labelsize=16)
    # ax.legend(fontsize=14, loc='lower left', title_fontsize=14)
    
    plt.legend(loc='center left', title='model', bbox_to_anchor=(1, 0.5), title_fontsize=14, fontsize=14)

    # ax.legend(loc='upper right', bbox_to_anchor=(1.5, 0.6))
    sns.set_style("ticks")
    return fig, ax

def plot_parameters_kfold(df, n_cols_legend=2, figsize=(5.5, 5.5), hour_label=False, style='whitegrid',
                          rotation_xticks=0, **kwargs):
    # bbox_to_anchor_utility = [0.27, -0.15]
    sns.set_style(style)
    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=figsize)
    bbox_to_anchor = [0.55, -0.27]
    x_label = 'period'
    if hour_label:
        x_label = 'hour'
        df.hour = pd.to_datetime(df['hour'], format='%H').dt.strftime('%I%p').str.lstrip("0")
    sns.pointplot(data=df, x=x_label, y='value', hue='parameter', ax=axs, **kwargs)
    axs.legend(loc='upper center',
               # ncols=len(df.query("group == 'utility'").parameter.unique()),
               fontsize=12,
               ncols=n_cols_legend,
               bbox_to_anchor=bbox_to_anchor,
               bbox_transform=BlendedGenericTransform(fig.transFigure, axs.transAxes))
    axs.set_xlabel(axs.get_xlabel(), fontsize=18)
    axs.set_ylabel('parameter value', fontsize=18)
    axs.tick_params(axis='both', which='major', labelsize=16)
    sns.set_style("ticks")
    plt.xticks(rotation=rotation_xticks)

    return fig, axs

def plot_parameters(*args, **kwargs):
    # bbox_to_anchor_utility = [0.27, -0.15]
    return plot_parameters_kfold(style='ticks', *args, **kwargs)

def plot_flow_interaction_matrix(flow_interaction_matrix,
                                 masking_matrix,
                                 links_ids=None, **kwargs):
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(9, 4), sharex=True, sharey=True)
    # fig, ax = plt.subplots(figsize=(5.5, 5), tight_layout=True)
    if not isinstance(flow_interaction_matrix, np.ndarray):
        flow_interaction_matrix = (flow_interaction_matrix * np.eye(masking_matrix.shape[0]))
    if links_ids is None:
        links_ids = np.array(range(1, flow_interaction_matrix.shape[0] + 1)).astype(str)
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    for matrix, ax in zip([masking_matrix, flow_interaction_matrix], axs):
        if kwargs.get('vmin', None) is None or kwargs.get('vmax', None) is None:
            bound = np.max(np.abs(flow_interaction_matrix))
            kwargs['vmin'], kwargs['vmax'] = -bound, bound
        matrix_df = pd.DataFrame({'link_1': pd.Series([], dtype=int),
                                  'link_2': pd.Series([], dtype=int),
                                  'weight': pd.Series([], dtype=float)})
        rows, cols = matrix.shape
        counter = 0
        for i, link_id1 in zip(range(0, rows), links_ids):
            for j, link_id2 in zip(range(0, cols), links_ids):
                matrix_df.loc[counter] = [int(link_id1), int(link_id2), matrix[(i, j)]]
                counter += 1
        matrix_df.link_1 = matrix_df.link_1.astype(int)
        matrix_df.link_2 = matrix_df.link_2.astype(int)
        matrix_pivot_df = matrix_df.pivot_table(index='link_1', columns='link_2', values='weight')
        # ax
        sns.heatmap(matrix_pivot_df, linewidth=0.5, cmap=cmap, ax=ax, **kwargs)
        
    for ax in axs:
        ax.set_xlabel('link', fontsize=16)
        ax.set_ylabel('link', fontsize=16)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation = 'horizontal')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
        ax.yaxis.set_tick_params(which='both', labelleft=True, rotation='auto', labelsize=12)
        ax.xaxis.set_tick_params(which='both', labelbottom=True, rotation=90, labelsize=12)
        ax.collections[0].colorbar.ax.tick_params(labelsize=12)
        # ax.tick_params(axis='x', rotation=90)
        # ax.tick_params(axis='y',rotation = 90)
        # ax.ticklabel_format(style = 'plain')
        # ax.locator_params(integer=True)
        # ax.get_yaxis().set_major_formatteryaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.yticks(rotation = 'horizontal')
    axs[0].set_title('Initial kernel', fontsize=18)
    axs[1].set_title('Final kernel', fontsize=18)
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: np.math.ceil(x)))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: np.math.ceil(x)))
    # plt.show()
    return fig, axs

def plot_predictive_performance(train_losses: pd.DataFrame,
                                val_losses: pd.DataFrame = None,
                                epochs_end_learning_stage: int = None,
                                xticks_spacing: int = 5,
                                show_validation=False,
                                show_equilibrium_stage_line=False,
                                curves=None,
                                prefix_metric='loss',
                                yaxis_label='relative mse',
                                show_percentage_units=False,
                                **kwargs):
    # fig, ax = plt.subplots(figsize = (5,4))
    if show_equilibrium_stage_line:
        fig, ax = plt.subplots(figsize=(5.5, 5), tight_layout=True)
    elif show_validation:
        fig, ax = plt.subplots(figsize=(5.5, 5), tight_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    if curves is None:
        curves = ['travel time', 'link flow', 'equilibrium']
    if epochs_end_learning_stage is not None and show_equilibrium_stage_line:
        ax.axvline(epochs_end_learning_stage, linestyle='dotted', color='black')
    patches = []
    if 'travel time' in curves:
        patches.append(mpatches.Patch(color='blue', label='link flow'))
        ax.plot(train_losses['epoch'], train_losses[prefix_metric + '_traveltime'], label="travel time", color='red',
                linestyle='-')
        if show_validation:
            ax.plot(val_losses['epoch'], val_losses[prefix_metric + '_traveltime'], label="travel time (val)",
                    color='red',
                    linestyle='--')
    if 'link flow' in curves:
        patches.append(mpatches.Patch(color='red', label='travel time'))
        ax.plot(train_losses['epoch'], train_losses[prefix_metric + '_flow'], label="link flow", color='blue',
                linestyle='-')
        if show_validation:
            ax.plot(val_losses['epoch'], val_losses[prefix_metric + '_flow'], label="link flow (val)", color='blue',
                    linestyle='--')
    if 'equilibrium' in curves and prefix_metric + '_equilibrium' in train_losses.columns:
        patches.append(mpatches.Patch(color='gray', label='equilibrium'))
        ax.plot(train_losses['epoch'], train_losses[prefix_metric + '_equilibrium'], label="equilibrium", color='gray',
                linestyle='-')
        if show_validation:
            ax.plot(val_losses['epoch'], val_losses[prefix_metric + '_equilibrium'], label="equilibrium (val)",
                    color='gray',
                    linestyle='--')
    if prefix_metric in ['loss']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    ticks = np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + xticks_spacing, xticks_spacing)
    ax.set_xticks(np.arange(train_losses['epoch'].min(), train_losses['epoch'].max() + 1, xticks_spacing))
    # ax.set_xlim(xmin=train_losses['epoch'].min(), xmax=train_losses['epoch'].max())
    ax.set_xlim(train_losses['epoch'].min() - (xticks_spacing * 0.2),
                train_losses['epoch'].max() + (xticks_spacing * 0.2))
    # plt.ylim(ymin=0, ymax=100)
    ax.set_ylim(ymin=0)
    ax.set_xlabel('epoch')
    # ax.set_ylabel('loss')
    ax.set_ylabel(yaxis_label)
    if show_equilibrium_stage_line:
        training_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='solid', label='training loss')
        validation_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dashed', label='validation loss')
        equilibrium_stage_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dotted',
                                            label='start of equilibrium stage')
        legend_size = 12
        handles = [training_line, equilibrium_stage_line]
        if show_validation:
            handles = [training_line, validation_line, equilibrium_stage_line]
            legend1 = plt.legend(handles=handles, loc='upper center',
                                 ncol=3,
                                 # , prop={'size': self.fontsize}
                                 bbox_to_anchor=[0.55, -0.18],
                                 bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                                 prop={'size': legend_size}
                                 )
        else:
            legend1 = plt.legend(handles=handles, loc='upper center',
                                 ncol=2,
                                 # , prop={'size': self.fontsize}
                                 bbox_to_anchor=[0.55, -0.18],
                                 bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                                 prop={'size': legend_size})
        ax.add_artist(legend1)
        legend2 = plt.legend(handles=patches, loc='upper center', ncol=len(patches),  # , prop={'size': self.fontsize}
                             bbox_to_anchor=[0.55, -0.26],
                             bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                             prop={'size': legend_size}
                             )
        # ax.add_artist(legend2)
        # legend2 = plt.legend(handles=[train_patch, val_patch], handleheight=1e-2, loc='upper center', ncol=2#, prop={'size': self.fontsize}
        #            , bbox_to_anchor=[0.52, -0.4]
        #            , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes))
        fig.subplots_adjust(top=0.95, bottom=0.28)
    elif show_validation:
        training_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='solid', label='training loss')
        validation_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='dashed', label='validation loss')
        legend_size = 12
        handles = [training_line]
        if show_validation:
            handles = [training_line, validation_line]
            legend1 = plt.legend(handles=handles, loc='upper center',
                                 ncol=2,
                                 # , prop={'size': self.fontsize}
                                 bbox_to_anchor=[0.55, -0.18],
                                 bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                                 prop={'size': legend_size}
                                 )
        else:
            legend1 = plt.legend(handles=handles, loc='upper center',
                                 ncol=1,
                                 # , prop={'size': self.fontsize}
                                 bbox_to_anchor=[0.55, -0.18],
                                 bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                                 prop={'size': legend_size})
        ax.add_artist(legend1)
        legend2 = plt.legend(handles=patches, loc='upper center', ncol=len(patches),  # , prop={'size': self.fontsize}
                             bbox_to_anchor=[0.55, -0.3],
                             bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes),
                             prop={'size': legend_size}
                             )
        # ax.add_artist(legend2)
        # legend2 = plt.legend(handles=[train_patch, val_patch], handleheight=1e-2, loc='upper center', ncol=2#, prop={'size': self.fontsize}
        #            , bbox_to_anchor=[0.52, -0.4]
        #            , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes))
        fig.subplots_adjust(top=0.95, bottom=0.28)
    else:
        # Showing validation curve and equilibrium line
        ax.legend(loc="upper center", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    if not show_validation:
        ax.legend(fontsize=14, title_fontsize=14)

    if show_percentage_units:
        percent_formatter = mtick.PercentFormatter(xmax=100, decimals=0)
        plt.gca().yaxis.set_major_formatter(percent_formatter)

    plt.tight_layout()
    return fig, ax

def plot_annotate_r2(ax, x, y, intercept=False, all_metrics=False, r2=False, rho=False):
    nas = np.logical_or(np.isnan(x.values), np.isnan(y.values))
    # slope, intercept, r, p, se = sp.stats.linregress(x = x[~nas], y = y[~nas])
    # r2 = r**2
    # # ax = plt.gca()
    # _, _, rho_val, _, _ = sp.stats.linregress(x=x[~nas], y=y[~nas])
    rho_val, p = sp.stats.pearsonr(x[~nas], y[~nas])
    reg = LinearRegression(fit_intercept=intercept).fit(X=x[~nas].values.reshape(-1, 1),
                                                        y=y[~nas].values.reshape(-1, 1))
    r2_val = r2_score(x, y)
    intercept_value = float(reg.intercept_)
    slope = float(reg.coef_)
    if rho and not r2:
        ax.text(.05, .9, r'$\rho$={:.2f}'.format(rho_val), transform=ax.transAxes)
    if r2 and not rho:
        # ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}'.format(r2, r), transform=ax.transAxes)
        ax.text(.05, .9, r'$R^2$={:.2f}'.format(r2_val), transform=ax.transAxes)
    if r2 and rho:
        ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}'.format(r2_val, rho_val), transform=ax.transAxes)
    if all_metrics:
        # ax.text(.05, .8, r'$R^2$={:.2f}, $\beta$={:.2f}, p={:.2f}'.format(r2, slope,p),
        #         transform=ax.transAxes)
        slope, intercept, r, p, se = sp.stats.linregress(x=x[~nas], y=y[~nas])
        ax.text(.05, .9, r'$R^2$={:.2f}, $\rho$={:.2f}, $\beta$={:.2f}, p={:.2f}'.format(r2_val, r, slope, p),
                transform=ax.transAxes)
    new_x = np.concatenate([np.array([0]), x])
    ax.plot(new_x, intercept_value + slope * new_x, '-', color='black')

def plot_observed_flow_vs_traveltime(model, period_col=None, observed_traveltime=None, observed_flow=None,
                                     hour_label=False, all_metrics=False, **kwargs):
    hue = 'period'
    # hue = 'sign'
    if hour_label:
        hue = 'hour'
    if period_col is None:
        period_col = 1
        hue = None
    plot_data = pd.DataFrame({hue: period_col})
    plot_data[hue] = pd.Categorical(plot_data[hue],
                                    plot_data[hue].sort_values().unique())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True,
                           # sharex=True, sharey=True
                           )
    observed_flow = model.mask_observed_flow(observed_flow).numpy().flatten()
    observed_traveltime = model.mask_observed_traveltime(observed_traveltime).numpy().flatten()
    plot_data['observed_traveltime'] = observed_traveltime
    plot_data['observed_flow'] = observed_flow
    sns.scatterplot(data=plot_data, x='observed_flow', y='observed_traveltime', hue=hue, ax=ax, **kwargs)
    ax.set_xlabel('observed flow', fontsize=16)
    ax.set_ylabel('observed travel time', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plot_annotate_r2(ax=ax, x=plot_data['observed_flow'], y=plot_data['observed_traveltime'],
                     all_metrics=all_metrics, intercept=False, rho=True)
    ax.margins(x=0.05, y=0.05)
    ax.legend(title=hue)
    return fig, ax

def plot_flow_vs_traveltime(model, period_col=None, observed_traveltime=None, observed_flow=None,
                            hour_label=False, all_metrics=False, only_observed=False, ampm_format=True, **kwargs):
    if only_observed:
        return plot_observed_flow_vs_traveltime(model=model, period_col=period_col,
                                                observed_traveltime=observed_traveltime, observed_flow=observed_flow,
                                                hour_label=hour_label, all_metrics=all_metrics, **kwargs)
    hue = 'period'
    # hue = 'sign'
    if hour_label:
        hue = 'hour'
    if period_col is None:
        period_col = 1
        hue = None
    plot_data = pd.DataFrame({'predicted_flow': model.predict_flow().numpy().flatten(),
                              'predicted_traveltime': model.predict_traveltime().numpy().flatten(),
                              hue: period_col
                              })
    plot_data[hue] = pd.Categorical(plot_data[hue],
                                    plot_data[hue].sort_values().unique())
    plot_data['sign'] = ((plot_data.predicted_flow >= 0) & (plot_data.predicted_traveltime >= 0)). \
        astype(int).map({0: 'inconsistent', 1: 'consistent'})
    if ampm_format:
        plot_data[hue] = plot_data[hue].astype(str).apply(lambda x: time.strftime("%l%p", time.strptime(x, "%H")))
    if observed_flow is None and observed_traveltime is None:
        fig, axs = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
        axs = sns.scatterplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', hue=hue, ax=axs)
        plot_annotate_r2(ax=axs, x=plot_data['predicted_flow'], y=plot_data['predicted_traveltime'],
                         all_metrics=all_metrics, intercept=False, rho=True)
        axs.set_xlabel(axs.get_xlabel(), fontsize=18)
        axs.set_ylabel(axs.get_ylabel(), fontsize=18)
        axs.tick_params(axis='both', which='major', labelsize=12)

    elif observed_flow is not None and observed_traveltime is not None:
        fig, axs = plt.subplots(2, 2, figsize=(9, 8), tight_layout=True,
                                # sharex=True, sharey=True
                                )
        observed_flow = model.mask_observed_flow(observed_flow).numpy().flatten()
        observed_traveltime = model.mask_observed_traveltime(observed_traveltime).numpy().flatten()
        plot_data['observed_traveltime'] = observed_traveltime
        plot_data['observed_flow'] = observed_flow
        # plot_data = plot_data.dropna()
        sns.scatterplot(data=plot_data, x='observed_flow', y='observed_traveltime', hue=hue, ax=axs[0, 0],
                        legend=False, **kwargs)
        axs[0, 0].set_xlabel('observed flow')
        axs[0, 0].set_ylabel('observed travel time')
        sns.scatterplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', hue=hue, ax=axs[0, 1],
                        legend=False, **kwargs)
        axs[0, 1].set_xlabel('estimated flow')
        axs[0, 1].set_ylabel('estimated travel time')
        sns.scatterplot(data=plot_data, x='observed_flow', y='predicted_flow', hue=hue, ax=axs[1, 0],
                        legend=False, **kwargs)
        axs[1, 0].set_xlabel('observed flow')
        axs[1, 0].set_ylabel('estimated flow')
        sns.scatterplot(data=plot_data, x='observed_traveltime', y='predicted_traveltime', hue=hue, ax=axs[1, 1],
                        legend=True,
                        **kwargs)
        axs[1, 1].set_xlabel('observed travel time')
        axs[1, 1].set_ylabel('estimated travel time')
        # axs[1,1].legend.set_visible(False)
        # sns.regplot(data=plot_data, x='observed_flow', y='observed_traveltime', ax=axs[0, 0], **kwargs)
        # sns.regplot(data=plot_data, x='predicted_flow', y='predicted_traveltime', ax=axs[0, 1], **kwargs)
        #
        # sns.regplot(data=plot_data, x='observed_flow', y='predicted_flow', ax=axs[1, 0], **kwargs)
        # sns.regplot(data=plot_data, x='observed_traveltime', y='predicted_traveltime', ax=axs[1, 1], **kwargs)
        plot_annotate_r2(ax=axs[0, 0], x=plot_data['observed_flow'], y=plot_data['observed_traveltime'],
                         all_metrics=all_metrics, intercept=False, rho=True)
        plot_annotate_r2(ax=axs[0, 1], x=plot_data['predicted_flow'], y=plot_data['predicted_traveltime'],
                         all_metrics=all_metrics, intercept=False, rho=True)
        plot_annotate_r2(ax=axs[1, 0], x=plot_data['observed_flow'], y=plot_data['predicted_flow'],
                         all_metrics=all_metrics, intercept=False, r2=True)
        plot_annotate_r2(ax=axs[1, 1], x=plot_data['observed_traveltime'], y=plot_data['predicted_traveltime'],
                         all_metrics=all_metrics, intercept=False, r2=True)
        # axs[1].yaxis.set_tick_params(which='both', labelleft=True)
        # axs[1].get_yaxis().get_label().set_visible(True)
        # ax.set_ylabel('predicted_traveltime')
        # plt.ylabel('predicted_traveltime')
        # axs[1].set_yticks(axs[0].get_yticks())
        for ax in axs.reshape(-1):
            # ax.get_legend().remove()
            # ax.set_xlim(xmin=0)
            # ax.set_ylim(ymin=0)
            ax.margins(x=0.05, y=0.05)
            ax.set_xlabel(ax.get_xlabel(), fontsize=16)
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=12)
            # ax.legend(title=hue)
            # if np.sum(plot_data.sign == 'inconsistent') == 0:
            #     ax.get_legend().remove()
            
        n_periods = len(plot_data[hue].unique())
        reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
        h, l = axs[1, 1].get_legend_handles_labels()
        ncol = 8
        axs[1, 1].get_legend().remove()
        legend = fig.legend(handles=reorder(h, ncol), labels=reorder(l, ncol), loc='upper center',
                            bbox_to_anchor=(0.5, 0.02), ncol=ncol, handlelength=0, columnspacing=1, handletextpad=0.4,
                            labelspacing=0.4, fontsize=14, title_fontsize = 14)
        # legend.set_bbox_to_anchor((1.05, 1))
        legend.set_title(hue)
        # legend(handlelength=0, handletextpad=0)
        plt.tight_layout()

    return fig, axs

def plot_performance_functions(model, network, flow_range=None, marginal=False, type='mlp',
                               selected_links=None, alpha=None, beta=None, sharey=False, **kwargs):
    if flow_range is None:
        flow_range = range(0, 20000, 100)
    traveltime_flow_df = pd.DataFrame({})
    type_pf = model.performance_function.type
    flows_shape = model.flows.numpy()[1, None].shape
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True, sharey=sharey)
    links_selected = True
    if selected_links is None:
        links_selected = False
        selected_links = range(flows_shape[1])
    capacities = np.array([link.bpr.k for link in network.links])
    free_flow_traveltimes = np.array([link.bpr.tf for link in network.links])
    if alpha is None:
        alpha = 0.15 * np.ones_like(capacities)
        beta = 4 * np.ones_like(capacities)
    if marginal:
        # Link-wise marginal increase
        marginal_increase_df = pd.DataFrame({})
        for link_idx in selected_links:
            for i in flow_range:
                flows = np.zeros(flows_shape)
                flows[0, link_idx] = i
                traveltime_pf = model.traveltimes(
                    flows=np.repeat(flows, model._flows.numpy().shape[0], 0))[0, :].numpy()[link_idx]
                traveltime_bpr = bpr_function(flows=flows[0, link_idx],
                                              free_flow_traveltimes=free_flow_traveltimes[link_idx],
                                              capacities=capacities[link_idx],
                                              alpha=alpha[link_idx],
                                              beta=beta[link_idx]
                                              ).numpy().flatten()
                traveltime_bpr = model.mask_predicted_traveltimes(traveltime_bpr, k=capacities[link_idx])
                marginal_increase_df = pd.concat([marginal_increase_df,
                                                  pd.DataFrame({'flow': i,
                                                                'link': link_idx,
                                                                'traveltime_exogenous_bpr': traveltime_bpr,
                                                                'traveltime_' + str(type_pf): traveltime_pf
                                                                })])
        if links_selected:
            marginal_increase_df['link'] = pd.Categorical(marginal_increase_df['link'],
                                                          marginal_increase_df['link'].sort_values().unique())

        sns.lineplot(data=marginal_increase_df, x='flow', y='traveltime_' + str(type_pf), hue='link', ax=axs[0],
                     **kwargs)
        sns.lineplot(data=marginal_increase_df, x='flow', y='traveltime_exogenous_bpr', hue='link', ax=axs[1],
                     **kwargs)
        axs[0].set_title(type_pf, pad=20, fontsize=18)
        axs[1].set_title(
            f'exogenous bpr (alpha = {round(float(np.mean(alpha)), 2)}, beta = {round(float(np.mean(beta)), 2)})',
            pad=20, fontsize=18)
        for ax in axs:
            ax.set_xlabel('traffic flow', fontsize=16)
            ax.set_ylabel('travel time', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.yaxis.set_tick_params(which='both', labelleft=True, rotation='auto')
            ax.legend(fontsize=12, loc='upper left', title = 'link', title_fontsize=12)
        axs[1].axes.get_yaxis().get_label().set_visible(True)
        return fig
    # Joint marginal increase
    for i in flow_range:
        flows = i * np.ones(flows_shape)
        traveltime_pf = model.traveltimes(flows=np.repeat(flows, model._flows.numpy().shape[0], 0))[0, :].numpy()
        traveltime_bpr = bpr_function(flows=flows,
                                      free_flow_traveltimes=free_flow_traveltimes,
                                      capacities=capacities,
                                      alpha=alpha,
                                      beta=beta
                                      ).numpy().flatten()
        traveltime_bpr = model.mask_predicted_traveltimes(traveltime_bpr, k=capacities)
        traveltime_flow_df = pd.concat([traveltime_flow_df,
                                        pd.DataFrame({'flow': i,
                                                      'link': np.arange(network.get_n_links()),
                                                      'traveltime_exogenous_bpr': traveltime_bpr,
                                                      'traveltime_' + str(type_pf): traveltime_pf
                                                      })])
    plot_data = traveltime_flow_df[traveltime_flow_df.link.isin(selected_links)]
    if links_selected:
        plot_data['link'] = pd.Categorical(plot_data['link'], plot_data['link'].sort_values().unique())

    sns.lineplot(data=plot_data, x='flow', y='traveltime_' + str(type_pf), hue='link', ax=axs[0], **kwargs)
    axs[0].set_title(type_pf, pad=20, fontsize=18)
    sns.lineplot(data=plot_data, x='flow', y='traveltime_exogenous_bpr', hue='link', ax=axs[1], **kwargs)
    axs[1].set_title(
        f'exogenous bpr (alpha = {round(float(np.mean(alpha)), 2)}, beta = {round(float(np.mean(beta)), 2)})', pad=20, fontsize=18)
    for ax in axs:
        ax.yaxis.set_tick_params(which='both', labelleft=True, rotation='auto')
        ax.set_xlabel('traffic flow', fontsize=16)
        ax.set_ylabel('travel time', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12, loc='upper left', title = 'link', title_fontsize=12)
    axs[1].axes.get_yaxis().get_label().set_visible(True)
    
    return fig, axs

def plot_convergence_estimates(estimates: pd.DataFrame,
                               true_values: Dict = None,
                               xticks_spacing: int = 5):
    # # Add vot
    # estimates = estimates.assign(vot=true_values.apply(compute_vot, axis=1))
    estimates = pd.melt(estimates, ['epoch'], var_name='parameter')
    # #Add vot
    # true_values = true_values.assign(vot=true_values.apply(compute_vot, axis=1))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:len(estimates['parameter'].unique())]
    fig, ax = plt.subplots(nrows=1, ncols=1, tight_layout=True, figsize=(5.5, 5))
    if estimates['value'].notnull().sum() > 0:
        g = sns.lineplot(data=estimates, x='epoch', hue='parameter', y='value', ax=ax)
    if true_values is not None:
        true_values = pd.Series(true_values).to_frame().T
        true_values = true_values[estimates['parameter'].unique()]
        ax.hlines(y=true_values.values,
                  xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max(), colors=colors, linestyle='--')
    # ax.grid(False)
    # fig.set_size_inches(4, 3)
    plt.xticks(np.arange(estimates['epoch'].min(), estimates['epoch'].max() + 1, xticks_spacing))
    # plt.xlim(xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max() + 2)
    ax.set_xlim(estimates['epoch'].min() - (xticks_spacing * 0.2),
                estimates['epoch'].max() + (xticks_spacing * 0.2))
    
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=14, loc='upper left', title_fontsize=14)

    plt.tight_layout()
    return fig, ax

def plot_heatmap_demands(Qs: Dict[str, Matrix],
                         subplots_dims: Tuple,
                         figsize: Tuple,
                         vmin=None,
                         vmax=None,
                         folderpath: str = None,
                         filename: str = None) -> None:
    """
    Modification of heatmap_demand function from isuelogit package
    Assume list 'Qs' has 4 elements
    """
    fig, ax = plt.subplots(*subplots_dims, figsize=figsize)
    for Q, title, axi in zip(Qs.values(), Qs.keys(), ax.flat):
        rows, cols = Q.shape
        od_df = pd.DataFrame({'origin': pd.Series([], dtype=int),
                              'destination': pd.Series([], dtype=int),
                              'trips': pd.Series([], dtype=float)})
        counter = 0
        for origin in range(0, rows):
            for destination in range(0, cols):
                # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
                od_df.loc[counter] = [int(origin + 1), int(destination + 1), Q[(origin, destination)]]
                counter += 1
        od_df.origin = od_df.origin.astype(int)
        od_df.destination = od_df.destination.astype(int)
        od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')
        # uniform_data = np.random.rand(10, 12)
        sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues", vmin=vmin, vmax=vmax, ax=axi)
        axi.set_xlabel(axi.get_xlabel(), fontsize=16)
        axi.set_ylabel(axi.get_ylabel(), fontsize=16)
        axi.tick_params(axis='both', which='major', labelsize=12)
        axi.set_title(title, fontsize=18)
    plt.tight_layout()
    # plt.show()
    # fig.savefig(folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight")
    # plt.close(fig)
    return fig, ax

def plot_top_od_flows_periods(model, period_feature, period_keys, reference_od, top_k=10, join_points=False,
                              rotation_xticks=0, ampm_format=True):
    """
    Plot top od pairs according to the largest number of trips reported in historic OD matrix
    """
    q_df = pd.DataFrame({})
    period_dict = {v: k for k, v in model.period_dict.items()}
    for i in range(model.q.shape[0]):
        # q_dict = dict(zip(fresno_network.ods, list(tvodlulpe.q[i].numpy())))
        q_dict = dict(zip(model.triplist, list(model.q[i].numpy())))
        label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
        # label_period_feature_2 = label_period_feature_1+1
        # label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
        label_period_feature = label_period_feature_1
        if ampm_format:
            if label_period_feature > 12:
                label_period_feature = str(label_period_feature - 12) + 'PM'
            else:
                label_period_feature = str(label_period_feature) + 'AM'
        q_df = pd.concat([q_df, pd.DataFrame(q_dict, index=[label_period_feature])])
    q_df = q_df.transpose()
    if reference_od is not None:
        q_df.insert(loc=0, column='reference_od', value=reference_od)
    # top_q = q_df.loc[q_df.var(axis = 1).sort_values(ascending=False)[0:top_k].index].sort_index()
    top_q = q_df.loc[q_df['reference_od'].sort_values(ascending=False)[0:top_k].index]  # .sort_index()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
    sns.heatmap(top_q, linewidth=0.5, cmap="Blues", vmin=0, ax=ax)
    plt.xlabel(period_feature, fontsize=12)
    plt.ylabel('od pair', fontsize=12)
    # plt.show()
    # Plot total trips by hour
    if reference_od is not None:
        total_trips_by_hour = q_df.sum(axis=0)[1:]
    else:
        total_trips_by_hour = q_df.sum(axis=0)
    total_trips_by_hour = total_trips_by_hour.reset_index().rename(columns={'index': period_feature, 0: 'total_trips'})
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), tight_layout=True)
    if total_trips_by_hour.shape[0] > 1:
        g = sns.pointplot(data=total_trips_by_hour, x=period_feature, y='total_trips', ax=ax,
                          label='estimated ODs', join=join_points)
    else:
        g = sns.pointplot(data=total_trips_by_hour, x=period_feature, y='total_trips', ax=ax,
                          join=join_points)
        g.axhline(total_trips_by_hour['total_trips'].values[0], label='estimated od', linestyle='solid')
    if reference_od is not None:
        g.axhline(q_df.sum(axis=0)['reference_od'], label='reference OD', linestyle='dashed')
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel('total trips', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.xticks(rotation=rotation_xticks)
    ax.legend(fontsize=14, loc='upper left', title_fontsize=14)

    # plt.show()
    # plt.xlabel(period_feature, fontsize=12)
    # plt.ylabel('od pair', fontsize=12)
    return top_q, total_trips_by_hour

def plot_rr_by_period_models(models, period_keys, period_feature='hour', rotation_xticks=0, **kwargs):
    rr_by_hour_models = []
    for model_key, model in models.items():
        theta_df = pd.DataFrame({})
        period_dict = {v: k for k, v in model.period_dict.items()}
        for i in range(model.theta.shape[0]):
            theta_dict = dict(zip(model.utility.features, list(model.theta[i].numpy())))
            label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
            # label_period_feature_2 = label_period_feature_1 + 1
            # label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
            label_period_feature = label_period_feature_1
            if label_period_feature > 12:
                label_period_feature = str(label_period_feature - 12) + 'PM'
            else:
                label_period_feature = str(label_period_feature) + 'AM'
            theta_dict[period_feature + '_id'] = label_period_feature
            theta_dict[period_feature] = label_period_feature_1
            theta_df = pd.concat([theta_df, pd.DataFrame(theta_dict, index=[label_period_feature])])
        rr_df = theta_df.assign(rr=theta_df.apply(compute_rr, axis=1)).reset_index()[['rr', period_feature,
                                                                                      period_feature + '_id']]
        rr_df['model'] = model_key
        rr_by_hour_models.append(rr_df)
    rr_by_hour_models = pd.concat(rr_by_hour_models)
    rr_by_hour_models = rr_by_hour_models.sort_values([period_feature], ascending=True).reset_index().drop(
        columns={'index'})
    # rr_by_hour_models['model'] = pd.Categorical(rr_by_hour_models['model'], ['lue', 'odlue', 'odlulpe', 'tvodlulpe'])
    fig, ax = plt.subplots(figsize=(5.5, 5), tight_layout=True)
    sns.pointplot(data=rr_by_hour_models, x=period_feature + '_id', y="rr", ax=ax,
                  hue='model',
                  # markers=['o', 'v', 's', '+'],
                  # palette=["black", "black", "black", "black"],
                  **kwargs)
    
    
    ax.set_xlabel(period_feature, fontsize=18)
    ax.set_ylabel('reliability ratio', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.xticks(rotation=rotation_xticks)
    ax.legend(fontsize=16, loc='upper right', title_fontsize=14)

    # ax.legend()
    # plt.legend(loc='upper left')
    # ax.get_legend().remove()
    return rr_by_hour_models

def plot_rr_by_period(model, period_keys, rotation_xticks=0, model_key='', period_feature='hour'):
    plot_rr_by_period_models({model_key: model}, rotation_xticks=rotation_xticks,
                             period_keys=period_keys, period_feature=period_feature)

def compute_total_trips_models(models, period_feature, period_keys):
    """
    Plot total trips among hours
    """
    total_trips_by_hour_models = []
    for model_key, model in models.items():
        period_dict = {v: k for k, v in model.period_dict.items()}
        q_df = pd.DataFrame({})
        for i in range(model.q.shape[0]):
            # q_dict = dict(zip(fresno_network.ods, list(tvodlulpe.q[i].numpy())))
            q_dict = dict(zip(model.triplist, list(model.q[i].numpy())))
            label_period_feature_1 = int(period_keys[period_keys.period_id == period_dict[i]][period_feature].iloc[0])
            label_period_feature_2 = label_period_feature_1 + 1
            label_period_feature = f"{label_period_feature_1}-{label_period_feature_2}"
            # label_period_feature = label_period_feature_1
            q_df = pd.concat([q_df, pd.DataFrame(q_dict, index=[label_period_feature])])
        q_df = q_df.transpose()
        total_trips_by_hour = q_df.sum(axis=0).reset_index().rename(columns={'index': period_feature, 0: 'total_trips'})
        total_trips_by_hour['model'] = model_key
        total_trips_by_hour_models.append(total_trips_by_hour)
    total_trips_by_hour_models = pd.concat(total_trips_by_hour_models)
    total_trips_by_hour_models['order'] = total_trips_by_hour_models[period_feature].str.split('-').str[0].astype(int)
    total_trips_by_hour_models = total_trips_by_hour_models.sort_values('order', ascending=True)
    # total_trips_by_hour_models['model'] = pd.Categorical(total_trips_by_hour_models['model'],
    #                                                      ['lue', 'odlue', 'odlulpe', 'tvodlulpe'])
    return total_trips_by_hour_models

def plot_total_trips_models(models, period_feature, period_keys, reference_od: np.array = None, rotation_xticks=0,
                            **kwargs):
    total_trips_by_hour_models = compute_total_trips_models(models=models, period_feature=period_feature,
                                                            period_keys=period_keys)
    # Replace hours to AM/PM format
    total_trips_by_hour_models[period_feature] = total_trips_by_hour_models[period_feature].str.split('-').str[
        0].astype(str). \
        apply(lambda x: time.strftime("%l%p", time.strptime(x, "%H")))
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    g = sns.pointplot(data=total_trips_by_hour_models, x=period_feature, y='total_trips', ax=ax,
                      hue='model',
                      # markers=['o', 'v', 's', '+'], palette=["black", "black", "black", "black"],
                      **kwargs)
    if reference_od is not None:
        g.axhline(np.sum(reference_od), linestyle='dashed', color='black', label='reference OD')  #
    plt.ylabel('total trips', fontsize=12)
    ax.legend()
    ax.set_xlabel(period_feature, fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(rotation=rotation_xticks)
    ax.legend(fontsize=12, loc='lower right', title_fontsize=12)
    # plt.legend(loc='lower left')
    
    return total_trips_by_hour_models

def plot_utility_parameters_periods(model, period_keys, period_feature, include_vot=False, plot=True):
    theta_df = utility_parameters_periods(
        model, period_keys=period_keys, period_feature=period_feature, include_vot=include_vot)
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    bound = np.nanmax(theta_df[[i for i in theta_df.columns if i != period_feature]].abs().values)
    if plot:
        fig, ax = plt.subplots(tight_layout=True)
        sns.heatmap(theta_df[[i for i in theta_df.columns if i != period_feature]].transpose(),
                    linewidth=0.5, cmap=cmap,
                    vmin=-bound, vmax=bound, ax=ax)
        plt.xlabel(period_feature, fontsize=12)
        plt.ylabel('parameter', fontsize=12)
        # plt.show()
    return theta_df

def plot_congestion_maps(model, model_df: pd.DataFrame, gdf: gpd.GeoDataFrame, features: List[str], benchmark_df,
                         cmap='viridis', benchmark_model=None):

    if benchmark_model is None:
        benchmark_model = 'regression_kriging'
    # Train benchmark model
    X, y = get_tensors_by_year(benchmark_df, features_Z=features, links_keys=benchmark_df.link_key.drop_duplicates())
    X_train, y_train = X[2019], y[2019]
    X_val, y_val = X[2020], y[2020]
    # Provide centroids of links as geographical information to compute regression kriging.
    centroids = gdf.to_crs(2228).geometry.centroid.to_crs(4326)
    gdf['X'], gdf['Y'] = centroids.x, centroids.y
    # Prediction travel time
    _, predictions_traveltime_baseline = compute_baseline_predictions(X_train=X_train[0].numpy()[:, :1],
                                                                      X_val=X_val[0].numpy()[:, :1],
                                                                      y_train=y_train[0][:, 0].numpy(),
                                                                      y_val=y_val[0][:, 0].numpy(),
                                                                      coordinates_train=gdf[['X', 'Y']].values,
                                                                      coordinates_val=gdf[['X', 'Y']].values,
                                                                      models=[benchmark_model]
                                                                      )
    # Prediction traffic flow
    _, predictions_flow_baseline = compute_baseline_predictions(X_train=X_train[0].numpy(),
                                                                X_val=X_val[0].numpy(),
                                                                y_train=y_train[0][:, 1].numpy(),
                                                                y_val=y_val[0][:, 1].numpy(),
                                                                coordinates_train=gdf[['X', 'Y']].values,
                                                                coordinates_val=gdf[['X', 'Y']].values,
                                                                models=[benchmark_model]
                                                                )

    benchmark_df = benchmark_df[benchmark_df.year == 2020]. \
        assign(pred_traveltime_benchmark=list(predictions_traveltime_baseline.values())[0],
               pred_flow_benchmark=list(predictions_flow_baseline.values())[0])
    benchmark_df['pred_speed_benchmark'] = benchmark_df.eval(
        'length/pred_traveltime_benchmark') * 60  # Map to speed per hour
    # Prepare model dataset
    X, y = get_tensors_by_year(model_df, features_Z=features, links_keys=gdf.link_key)
    X, Y = X[2020], y[2020]
    # Add observed travel time and observed flow
    model_df['obs_traveltime'] = Y[:, :, 0].numpy().flatten()
    model_df['obs_flow'] = Y[:, :, 1].numpy().flatten()
    # Add predicted values
    model_df['pred_traveltime'] = model.predict_traveltime().numpy().flatten()
    model_df['pred_flow'] = model.predict_flow().numpy().flatten()
    # Select predictions that are within the date and hour of the benchmark dataset in 2020, i.e., the prediction year
    model_df = model_df[(model_df.year == 2020) & (model_df.period.isin(benchmark_df.period.unique()))]
    # Add features
    model_df['pred_speed'] = model_df.eval('length/pred_traveltime') * 60  # To map to speed per hour
    model_df['obs_speed'] = model_df.eval('length/obs_traveltime') * 60  # To map to speed per hour
    model_df.loc[model_df['obs_speed'] == float('inf'), 'obs_speed'] = float('nan')
    # Merge predictions from benchmark and model and then add spatial information
    plot_df = gpd.GeoDataFrame(pd.merge(pd.merge(benchmark_df.drop(columns=['capacity [veh]', 'speed_max']),
                                                 model_df, on='link_key'),
                                        gdf, on=['link_key']), geometry='geometry').to_crs(epsg=3857)

    # Create additional variables 
    plot_df['capacity'] = np.where(plot_df['capacity [veh]'] == float('inf'), float('nan'), plot_df['capacity [veh]'])
    plot_df['pred_flow_ratio'] = plot_df.eval('pred_flow/capacity')
    plot_df['obs_flow_ratio'] = plot_df.eval('obs_flow/capacity')
    plot_df['pred_flow_ratio_benchmark'] = plot_df.eval('pred_flow_benchmark/capacity')

    plot_df.loc[plot_df.speed_max == 0, 'speed_max'] = plot_df[plot_df.speed_max != 0].speed_max.mean()
    plot_df['obs_speed_ratio'] = plot_df.eval('obs_speed/speed_max')
    plot_df.loc[plot_df['pred_speed'] == float('inf'), 'pred_speed'] = float('nan')
    plot_df['pred_speed_ratio'] = plot_df.eval('pred_speed/speed_max')
    # Post-processing to ensure that predictions from benchmark are within reasonable range
    plot_df['pred_speed_ratio_benchmark'] = plot_df['pred_speed_benchmark'] / plot_df['speed_max']
    # plot_df['pred_speed_ratio_benchmark'] = plot_df['pred_speed_ratio_benchmark'].clip(0, plot_df['obs_speed_ratio'].max())
    # To show as percentage
    plot_df['obs_speed_ratio'] = 100 * plot_df['obs_speed_ratio']
    plot_df['pred_speed_ratio'] = 100 * plot_df['pred_speed_ratio']
    plot_df['pred_speed_ratio_benchmark'] = 100 * plot_df['pred_speed_ratio_benchmark']
    plot_df['obs_flow_ratio'] = 100 * plot_df['obs_flow_ratio']
    plot_df['pred_flow_ratio'] = 100 * plot_df['pred_flow_ratio']
    plot_df['pred_flow_ratio_benchmark'] = 100 * plot_df['pred_flow_ratio_benchmark']
    # Plot of Speed prediction
    # classifier_ = Natural_Breaks(data_values, k=5)
    num_classes = 4
    # data_values = plot_df[['obs_speed_ratio', 'pred_speed_ratio','pred_speed_ratio_benchmark']].dropna().values.flatten()
    # data_values = plot_df[['obs_speed_ratio']].dropna().values.flatten()
    # min_val, max_val = min(data_values), max(data_values)
    min_val, max_val = 0, 100
    breaks = np.linspace(min_val, max_val, num_classes + 1).tolist()
    # breaks = [data_values.quantile(i / num_classes) for i in range(1, num_classes)]
    # breaks.insert(0,min_val )  # Add the minimum value as the first break
    # breaks.append(max_val) # Add the maximum value as the last break
    show_legend = False
    scheme = 'userdefined'
    # scheme = 'UserDefined'
    # scheme = 'natural_breaks'
    # scheme = "equal_interval"
    # cmap = "viridis"
    # cmap = "plasma"
    # cmap = "OrRd"
    # cmap = "YlOrRd"
    # cmap = "Reds"
    fig_speed, axs = plt.subplots(1, 4, tight_layout=True, figsize=(8, 7), sharex=True, sharey=True)
    plot_df.plot(ax=axs[0], color='gray')
    ctx.add_basemap(source=ctx.providers.OpenStreetMap.Mapnik, ax=axs[0])
    axs[0].set_title('Network')
    plot_df[plot_df['obs_speed_ratio'] > 0].plot(column='obs_speed_ratio', scheme=scheme, cmap=cmap, ax=axs[1],
                                                 classification_kwds={'bins': breaks}, legend=show_legend,
                                                 legend_kwds={"fmt": "{:.0f}%"})
    axs[1].set_title('Ground Truth')
    plot_df[(plot_df['pred_speed_ratio'] > 0)].plot(column='pred_speed_ratio', scheme=scheme, cmap=cmap, ax=axs[2],
                                                    classification_kwds={'bins': breaks}, legend=show_legend,
                                                    legend_kwds={"fmt": "{:.0f}%"})
    axs[2].set_title('MaTE')
    plot_df[(plot_df['pred_speed_ratio_benchmark'] > 0)].plot(column='pred_speed_ratio_benchmark',
                                                              scheme=scheme, cmap=cmap, ax=axs[3],
                                                              classification_kwds={'bins': breaks}, legend=True,
                                                              legend_kwds={"fmt": "{:.0f}%"})
    axs[3].set_title('Kriging Regression')
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    legend = axs[3].get_legend()
    # Get rid of first interval with -infinity
    handles = legend.legendHandles[1:]
    labels = [text.get_text() for text in legend.texts[1:]]
    labels[0] = f"< {[x.strip() for x in labels[1].split(',')][0]}"
    labels[-1] = '> 100%'
    axs[3].legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(2.35, 0.6), title = 'ratio')
    fig_speed.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Ratio between predicted and maximum speed', x=0.4, y=0.97)
    # plt.show()
    # fig_speed.text(0.5, 0.95, 'Speed ratio', ha='center', fontsize=16)
    plt.close()
    # Plot of traffic flow prediction
    # scheme = 'natural_breaks'
    # scheme = "equal_interval"
    fig_flow, axs = plt.subplots(1, 4, tight_layout=True, figsize=(8, 7), sharex=True, sharey=True)
    plot_df.plot(ax=axs[0], color='gray')
    ctx.add_basemap(source=ctx.providers.OpenStreetMap.Mapnik, ax=axs[0])
    axs[0].set_title('Network')
    plot_df[(plot_df['obs_flow_ratio'] > 0)].plot(column='obs_flow_ratio', scheme=scheme, cmap=cmap, ax=axs[1],
                                                  classification_kwds={'bins': breaks}, legend=show_legend,
                                                  legend_kwds={"fmt": "{:.0f}%"})
    axs[1].set_title('Ground truth')
    plot_df[(plot_df['pred_flow_ratio'] > 0)].plot(column='pred_flow_ratio', scheme=scheme, cmap=cmap, ax=axs[2],
                                                   classification_kwds={'bins': breaks}, legend=show_legend,
                                                   legend_kwds={"fmt": "{:.0f}%"})
    axs[2].set_title('MaTE')
    plot_df[(plot_df['pred_flow_ratio_benchmark'] > 0)].plot(
        column='pred_flow_ratio_benchmark', scheme=scheme, cmap=cmap, ax=axs[3],
        classification_kwds={'bins': breaks}, legend=True, legend_kwds={"fmt": "{:.0f}%"})
    axs[3].set_title('Kriging Regression')
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    # fig = plt.gcf()
    # fig.tight_layout(rect=[0, 0, 0.8, 1])
    legend = axs[3].get_legend()
    # Get rid of first interval with -infinity
    handles = legend.legendHandles[1:]
    labels = [text.get_text() for text in legend.texts[1:]]
    labels[0] = f"< {[x.strip() for x in labels[1].split(',')][0]}"
    labels[-1] = '> 100%'
    axs[3].legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(2.35, 0.6), title = 'ratio')
    fig_flow.subplots_adjust(wspace=0, hspace=0)
    # fig_flow.text(0.5, 0.95, 'Flow ratio', ha='center', fontsize=16)
    plt.suptitle('Ratio between predicted  and maximum flow', x=0.4, y=0.97)
    plt.close()
    return fig_speed, fig_flow

def plot_relative_gap_by_period(model, period_keys):
    input_flow = model.input_flow
    output_flow = model.output_flow()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    relative_gaps = compute_relative_gap_by_period(input_flow=input_flow, output_flow=output_flow)
    plot_df = pd.DataFrame({'relative_gap': relative_gaps,
                            'hour': pd.DataFrame({'period_id': model.period_ids[:, 0]})['period_id'].map(
                                dict(zip(period_keys.period_id, period_keys.hour))).values})
    plot_df['hour'] = plot_df.hour.astype(str).apply(lambda x: time.strftime("%l%p", time.strptime(x, "%H")))
    sns.pointplot(data=plot_df, x='hour', y='relative_gap', ax=ax)
    plt.ylabel('relative gap', fontsize=12)
    plt.xticks(rotation=90)
    ax.set_ylim(ymin=0)
    plt.tight_layout()

def plot_equilibrium_loss_by_period(model, period_keys):

    input_flow = model.input_flow
    output_flow = model.output_flow()

    equilibrium_losses = \
        [mse(actual=input_flow[i, :], predicted=output_flow[i, :]).numpy() for i in range(output_flow.shape[0])]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    plot_df = pd.DataFrame({'equilibrium_loss': equilibrium_losses,
                            'hour': pd.DataFrame({'period_id': model.period_ids[:, 0]})['period_id'].map(
                                dict(zip(period_keys.period_id, period_keys.hour))).values})
    plot_df['hour'] = plot_df.hour.astype(str).apply(lambda x: time.strftime("%l%p", time.strptime(x, "%H")))
    sns.pointplot(data=plot_df, x='hour', y='equilibrium_loss', ax=ax)
    plt.ylabel('equilibrium loss', fontsize=12)
    plt.xticks(rotation=90)
    ax.set_ylim(ymin=0)
    plt.tight_layout()