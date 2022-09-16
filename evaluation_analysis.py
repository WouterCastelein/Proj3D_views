import os
import random
import time

import numpy as np
from PyQt5 import QtCore
from matplotlib.patches import Patch

import constants
import visualization_tool
import pyqtgraph as pg
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm


def get_concat_h(im1, im2):
    """
    Paste two images together horizontally
    """
    dst = Image.new('RGB', (im1.width + im2.width, im1.height), (255, 255, 255))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    """
    Paste two images together vertically
    """
    dst = Image.new('RGB', (im1.width, im1.height + im2.height), (255, 255, 255))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def save_sphere(avp):
    himages = []
    myFont = ImageFont.truetype('Roboto-Regular.ttf', 200)
    for dataset in list(avp.keys()):
        for index, projection in enumerate(avp[dataset]):
            images = []
            for metric in constants.metrics:
                if index == len(avp[dataset]) - 1:
                    height = 240
                else:
                    height = 152
                im1 = Image.open(f"{constants.analysis_dir}/{dataset}-{projection}-{metric}-sphere1.png")
                im1 = im1.crop(
                    (im1.width / 2 - 152, im1.height / 2 - 152, im1.width / 2 + 152, im1.height / 2 + height))
                im2 = Image.open(f"{constants.analysis_dir}/{dataset}-{projection}-{metric}-sphere2.png")
                im2 = im2.crop(
                    (im2.width / 2 - 152, im2.height / 2 - 152, im2.width / 2 + 152, im2.height / 2 + height))
                images.append(im1)
                images.append(im2)
            images.append(Image.new('RGB', (images[0].width * 2, images[0].height), (255, 255, 255)))
            images = [Image.new('RGB', (images[0].width * 3 + 40, images[0].height), (255, 255, 255))] + images
            image = get_concat_h(images[0], images[1])
            for i in range(2, len(images)):
                image = get_concat_h(image, images[i])

            I1 = ImageDraw.Draw(image)
            I1.text((sum([im.width for im in images[:len(images) - 1]]) + 40, 50), f'{projection}', font=myFont,
                    fill=(0, 0, 0))
            if index == 0:
                str = dataset
                if dataset == 'WisconsinBreastCancer':
                    str = 'WBC'
                I1.text((10, 20), str, font=myFont, fill=(0, 0, 0))
            himages.append(image)
    himages = [Image.new('RGB', (himages[0].width, images[0].height), (255, 255, 255))] + himages
    I2 = ImageDraw.Draw(himages[0])
    for index, metric in enumerate(constants.metrics):
        I2.text((sum([im.width for im in images[:len(images) - (len(constants.metrics) - index) * 2]]) - 50, 50),
                metric[0].capitalize(), font=myFont, fill=(0, 0, 0))
    big_image = get_concat_v(himages[0], himages[1])
    for i in range(2, len(himages)):
        big_image = get_concat_v(big_image, himages[i])
    big_image.show()
    big_image.save(f'spheres.png')

def save_sphere_and_histograms(avp):
    himages = []
    myFont = ImageFont.truetype('Roboto-Regular.ttf', 200)
    datasets = list(avp.keys())
    datasets.sort()
    for dataset in datasets:
        projections = list(avp[dataset])
        projections.sort()
        for index, projection in enumerate(projections):
            images = []
            for metric in constants.metrics:
                if index == len(avp[dataset]) - 1:
                    height = 240
                else:
                    height = 152
                im1 = Image.open(f"{constants.analysis_dir}/{dataset}-{projection}-{metric}-sphere1.png")
                im1 = im1.crop(
                    (im1.width / 2 - 152, im1.height / 2 - 152, im1.width / 2 + 152, im1.height / 2 + height))
                im2 = Image.open(f"{constants.analysis_dir}/{dataset}-{projection}-{metric}-sphere2.png")
                im2 = im2.crop(
                    (im2.width / 2 - 152, im2.height / 2 - 152, im2.width / 2 + 152, im2.height / 2 + height))
                images.append(im1)
                images.append(im2)
            hist_image = Image.open(f'{constants.analysis_dir}/{dataset}-{projection}-histograms.png')
            step = 101
            N = hist_image.crop((54, 26      , 48 + 580, 26 + step))
            S = hist_image.crop((54, 26 + step, 48 + 580, 26 + 2 * step))
            C = hist_image.crop((54, 26 + 2 * step, 48 + 580, 26 + 3 * step))
            T = hist_image.crop((54, 26 + 3 * step, 48 + 580, 26 + 4 * step))
            for im in [T, C, S, N]:
                images.append(im.resize((int(im.width * (304 / im.height)), 304), Image.ANTIALIAS))
            images.append(Image.new('RGB', (images[0].width * 2, images[0].height), (255, 255, 255)))
            images = [Image.new('RGB', (images[0].width * 3 + 40, images[0].height), (255, 255, 255))] + images
            image = get_concat_h(images[0], images[1])
            for i in range(2, len(images)):
                image = get_concat_h(image, images[i])

            I1 = ImageDraw.Draw(image)
            I1.text((sum([im.width for im in images[:len(images) - 1]]) + 40, 50), f'{projection}', font=myFont,
                    fill=(0, 0, 0))
            if index == 0:
                str = dataset
                if dataset == 'WisconsinBreastCancer':
                    str = 'WBC'
                I1.text((10, 20), str, font=myFont, fill=(0, 0, 0))
            himages.append(image)
    himages = [Image.new('RGB', (himages[0].width, images[0].height), (255, 255, 255))] + himages
    I2 = ImageDraw.Draw(himages[0])
    for index, metric in enumerate(constants.metrics):
        I2.text((sum([im.width for im in images[:10 - (len(constants.metrics) - index) * 2]]) - 50, 50),
                metric[0].capitalize(), font=myFont, fill=(0, 0, 0))
        I2.text((sum([im.width for im in images[:len(images) - (len(constants.metrics) - index)]]) - 900, 50),
                metric[0].capitalize(), font=myFont, fill=(0, 0, 0))
    big_image = get_concat_v(himages[0], himages[1])
    for i in range(2, len(himages)):
        big_image = get_concat_v(big_image, himages[i])
    big_image.show()
    big_image.save(f'spheresAndHistograms.png')

def save_box_plots():
    myFont = ImageFont.truetype('Roboto-Regular.ttf', 30)
    myFontsmall = ImageFont.truetype('Roboto-Regular.ttf', 25)
    for config in ['TC', 'SN']:
        himages = []
        for dataset, projection in constants.evaluation_set[1:]:
            images = []
            hist_image = Image.open(f'{constants.analysis_dir}/{dataset}-{projection}-boxplots2.png')
            step = 227.5
            y_offset = 108
            N = hist_image.crop((12, y_offset      , hist_image.width - 12, y_offset + step))
            S = hist_image.crop((12, y_offset + step, hist_image.width - 12, y_offset + 2 * step))
            C = hist_image.crop((12, y_offset + 2 * step, hist_image.width - 12, y_offset + 3 * step))
            T = hist_image.crop((12, y_offset + 3 * step, hist_image.width - 12, y_offset + 4 * step - 5))
            if config == 'TC':
                for im in [T, C]:
                    images.append(im)
            else:
                for im in [S, N]:
                    images.append(im)
            images.append(Image.new('RGB', (int(images[0].width * 0.16), images[0].height), (255, 255, 255)))
            images = [Image.new('RGB', (int(images[0].width * 0.25), images[0].height), (255, 255, 255))] + images
            image = get_concat_h(images[0], images[1])
            for i in range(2, len(images)):
                image = get_concat_h(image, images[i])
            I1 = ImageDraw.Draw(image)
            for index, text in enumerate(['users-guided', 'users-blind', 'histogram']):
                I1.text((image.width - 145, image.height - 30 * (index+1) - 40), f'{text}', font=myFontsmall,
                        fill=(0, 0, 0))
            I1.text((10, 60), f'{dataset}-{projection}', font=myFont, fill=(0, 0, 0))
            himages.append(image)
        himages = [Image.new('RGB', (himages[0].width, int(images[0].height / 2)), (255, 255, 255))] + himages
        I2 = ImageDraw.Draw(himages[0])
        if config == 'TC':
            for index, metric in enumerate(constants.metrics[:2]):
                I2.text((sum([im.width for im in images[:len(images) - (len(constants.metrics) -2 - index)]]) - 450, 20),
                        metric[0].capitalize(), font=myFont, fill=(0, 0, 0))
        else:
            for index, metric in enumerate(constants.metrics[2:]):
                I2.text((sum([im.width for im in images[:len(images) - (len(constants.metrics) -2 - index)]]) - 450, 20),
                        metric[0].capitalize(), font=myFont, fill=(0, 0, 0))
        big_image = get_concat_v(himages[0], himages[1])
        for i in range(2, len(himages)):
            big_image = get_concat_v(big_image, himages[i])
        big_image.show()
        big_image.save(f'boxplots{config}.png')

def parse_pickles(update_NS = True):
    """
    Parse evaluation data
    """
    evaluation_files = glob.glob('evaluationdata/*.pkl')
    data = []
    for file_name in evaluation_files:
        with open(file_name, 'rb') as file:
            data.append(pickle.load(file))
    df = pd.DataFrame(data[0])
    for entry in data[1:]:
        df = df.append(entry, ignore_index = True)
    if update_NS:
        #Update the normalized stress values since they where calculated wrongly in the user experiment
        consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
        df_consolid = pd.read_pickle(consolid_metrics)
        viewpoints = df['viewpoint']
        df = df.reset_index()
        vps = df['viewpoint'].values
        projections = df['projection_method'].values
        datasets = df['dataset'].values
        qualities_2d = []
        qualities_3d = []
        view_quality = []
        for index, vp in enumerate(vps):
            metric_data = df_consolid.loc[(df_consolid['projection_name'] == projections[index])
                                          & (df_consolid["dataset_name"] == f"{datasets[index]}.pkl")]
            metric_data2D = metric_data.loc[metric_data['n_components'] == 2]
            metric_data3D = metric_data.loc[metric_data['n_components'] == 3]
            qualities_2d.append(metric_data2D[constants.metrics].to_numpy()[0])
            qualities_3d.append(metric_data3D[constants.metrics].to_numpy()[0])

            # Find the viewpoint for which me have metrics. that is closest to the current viewpoint
            view_points = np.load(f'spheres/sphere{constants.samples}_points.npy')
            distances = np.sum((view_points - vp / np.sqrt(np.sum(vp**2))) ** 2, axis=1)
            nearest = np.argmin(distances)

            # Get the metric values, and highlight the corresponding histogram bars
            nearest_values = metric_data3D['views_metrics'].values[0][nearest][1:]
            view_quality.append(nearest_values)
        df['2D_quality'] = qualities_2d
        df['3D_quality'] = qualities_3d
        df['view_quality'] = view_quality
    return df

def preference_bar_graph():
    """
    Create bargraph of whether users prefer 2D or 3D, for both blind and guided set
    """
    data = []
    labels = []
    pickles = parse_pickles()
    for dataset, projection in constants.evaluation_set[1:]:
        data_with_tools = pickles[(pickles['dataset'] == dataset) &
                                             (pickles['projection_method'] == projection) &
                                             (pickles['mode'] == 'eval_full')]
        percentage_with_tools  = 100 * len(data_with_tools[data_with_tools['preference'] == '3D']) / len(data_with_tools)
        data_without_tools = pickles[(pickles['dataset'] == dataset) &
                                                (pickles['projection_method'] == projection) &
                                                (pickles['mode'] == 'eval_half')]
        percentage_without_tools = 100 * len(data_without_tools[data_without_tools['preference'] == '3D']) / len(data_without_tools)
        data.append([percentage_without_tools, percentage_with_tools])
        labels.append(f'{dataset}-{projection}')
    data.append([sum(np.array(data)[:, 0]) / len(data) ,sum(np.array(data)[:, 1]) / len(data)])
    labels.append('Total')
    data = np.array(data)
    data = data.round(0)
    fig, ax = plt.subplots()
    X = np.arange(len(data))
    bar_without = ax.bar(X + 0.00, data[:, 0], color=(0, 0, 1, 0.7), width=0.33, label="Blind set")
    bar_with = ax.bar(X + 0.33, data[:, 1], color=(0, 1, 0, 0.7), width=0.33, label="Guided set")
    #ax.hlines(50, -1, 6, color=(1, 0, 0, 0.7))
    ax.set_ylabel('Percentage')
    ax.set_xticks(X + 0.33 / 2, labels)
    ax.legend()
    ax.bar_label(bar_without, padding=3)
    ax.bar_label(bar_with, padding=3)
    #fig.tight_layout()
    plt.show()

def metric_averages():
    """
    Compute average metric values for both 2D and 3D.
    """
    consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
    df_consolid = pd.read_pickle(consolid_metrics)
    df = df_consolid
    twoD =  df.loc[df['n_components'] == 2]
    threeD = df.loc[df['n_components'] == 3]
    metrics_2d = np.mean(twoD[constants.metrics].to_numpy(), axis=0)
    metrics_3d = np.mean(threeD[constants.metrics].to_numpy(), axis=0)
    print(f'2D: {metrics_2d}')
    print(f'3D: {metrics_3d}')

def viewpoints_that_beat_2d_all():
    """
    Create bar plot of the amount of views with higher quality than the 3D projection
    """

    df_consolid = constants.get_consolid_metrics()
    per_dataset = []
    datasets = list(set(df_consolid['dataset_name'].values))
    datasets.sort()
    projections = list(set(df_consolid['projection_name'].values))
    projections.sort()
    for dataset in datasets:
        per_projection = []
        for projection in projections:
            df = df_consolid.loc[(df_consolid['projection_name'] == projection)
                                 & (df_consolid['dataset_name'] == dataset)]
            twoD = df.loc[df['n_components'] == 2]
            threeD = df.loc[df['n_components'] == 3]
            metrics_2d = np.mean(twoD[constants.metrics].to_numpy(), axis=0)
            views_metrics = np.concatenate(threeD['views_metrics'].to_numpy())[:, 1:]
            better_views_indices = [np.where(views_metrics[:, i] > x) for i, x in enumerate(metrics_2d)]
            better_views = better_views_indices[0]
            for i in range(1, len(better_views_indices)):
                better_views = np.intersect1d(better_views, better_views_indices[i])
            if better_views.size > 0:
                print(dataset, projection)
            per_projection.append([metric.size for metric in better_views])
        per_dataset.append(per_projection)
    per_dataset = np.array(per_dataset)
    #labels = constants.metrics
    width = 0.23  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    x = np.arange(len(datasets))
    cmap = plt.get_cmap('tab10')
    colors = cmap(x)
    for i, metric in enumerate(constants.metrics):
        bottom = np.zeros(len(x))
        for j, projection in enumerate(projections):
            p = per_dataset[:, j, i]
            ax.bar(x + (width * (i - 1.5)), p, width, color=colors[j], bottom = bottom, edgecolor='white')
            if j == 0:
                bar = ax.bar(x + (width * (i - 1.5)), np.zeros(len(x)), width)
                ax.bar_label(bar, [metric[:1].upper() for _ in p], padding = -20)
            bottom += p
    ax.set_xticks(x, [d[:-4] if d != 'WisconsinBreastCancer.pkl' else 'WBC' for d in datasets])
    ax.tick_params(axis='x', which='major', pad=20)
    ax.set_ylabel('Number of views with higher metric values')
    legend_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label= projection) for i, projection in enumerate(projections)][::-1]
    ax.legend(handles=legend_elements)
    plt.show()

def p_values():
    #did the users pick significantly higher quality points than the average quality?
    data_users = parse_pickles()
    for dataset, projection in constants.evaluation_set[1:]:
        views_metrics = constants.get_views_metrics(dataset, projection)
        user_views_data = data_users.loc[(data_users['dataset'] == dataset) & (data_users['projection_method'] == projection)]
        user_views_metrics_without_tool = np.stack(user_views_data.loc[user_views_data['mode'] == 'eval_half']['view_quality'].to_numpy())
        user_views_metrics_with_tool = np.stack(user_views_data.loc[user_views_data['mode'] == 'eval_full']['view_quality'].to_numpy())
        user_views_metrics_total = np.stack(user_views_data['view_quality'].to_numpy())
        user_views_metrics_3D_preference = np.stack(
            user_views_data.loc[user_views_data['preference'] == '3D']['view_quality'].to_numpy())
        user_views_metrics_2D_preference = np.stack(
            user_views_data.loc[user_views_data['preference'] == '2D']['view_quality'].to_numpy())
        means = np.mean(user_views_metrics_with_tool, axis=0), \
                np.mean(user_views_metrics_without_tool, axis=0), \
                np.mean(user_views_metrics_total, axis=0), \
                np.mean(views_metrics, axis=0)
        stds = np.std(user_views_metrics_with_tool, axis=0), \
               np.std(user_views_metrics_without_tool, axis=0), \
               np.std(user_views_metrics_total, axis=0), \
               np.std(views_metrics, axis=0)
        t_test_without_tool = ttest_ind(user_views_metrics_without_tool, views_metrics, alternative='greater')
        t_test_with_tool = ttest_ind(user_views_metrics_with_tool, views_metrics, alternative='greater')
        t_test_total = ttest_ind(user_views_metrics_total, views_metrics, alternative='greater')
        t_test_3D_vs_2D = ttest_ind(user_views_metrics_3D_preference, user_views_metrics_2D_preference, alternative='greater')
        concat = np.array([t_test_without_tool.pvalue, t_test_with_tool.pvalue, t_test_total.pvalue, t_test_3D_vs_2D.pvalue])
        concat = concat.transpose().round(3)
        print(dataset, projection)
        print(t_test_3D_vs_2D.pvalue.round(3))

def correlation_3D_preference():
    """"
    Compute pearson rank correlation between whether view quality and whether users prefer 2D or 3D, for both sets.
    """
    user_data_all = parse_pickles()
    user_data_half = user_data_all.loc[user_data_all['mode'] == 'eval_half']
    user_data_full = user_data_all.loc[user_data_all['mode'] == 'eval_full']
    for selection in [user_data_half, user_data_full, user_data_all]:
        print('----------------')
        view_qualities = np.stack(selection['view_quality'])
        view_preference = [0 if p == '2D' else 1 for p in selection['preference']]
        for i in range(view_qualities.shape[1]):
            pearson_rank, p_value = pearsonr(view_qualities[:, i], view_preference)
            print(constants.metrics[i], str(pearson_rank.round(3)), p_value.round(3))

if __name__ == '__main__':
    #correlation_3D_preference()
    #p_values()
    #metric_averages()
    #viewpoints_that_beat_2d_all()

    #randomize evaluation set
    #evaluation_set = constants.evaluation_set[1:]
    #random.shuffle(evaluation_set)
    #constants.evaluation_set[1:] = evaluation_set

    #preference_bar_graph()

    # start tool
    data_frame = parse_pickles()
    win = visualization_tool.Tool(analysis_data=data_frame)
    win.showMaximized()
    pg.exec()

    #Save snapshots
    #QtCore.QTimer.singleShot(5000, lambda: win.save_user_selected_view_snapshots(0))

    # win.box_plot_images()
    #save_box_plots()

    # win.save_images((0, 0))
    # avp = win.available_datasets_projections()
    # save_sphere_and_histograms(avp)


