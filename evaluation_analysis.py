import random
import time

import numpy as np

import constants
import visualization_tool
import pyqtgraph as pg
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height), (255, 255, 255))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
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
            images.append(Image.new('RGB', (int(images[0].width * 0.15), images[0].height), (255, 255, 255)))
            images = [Image.new('RGB', (int(images[0].width * 0.25), images[0].height), (255, 255, 255))] + images
            image = get_concat_h(images[0], images[1])
            for i in range(2, len(images)):
                image = get_concat_h(image, images[i])
            I1 = ImageDraw.Draw(image)
            for index, text in enumerate(['users-full', 'users-half', 'histogram']):
                I1.text((image.width - 140, image.height - 30 * (index+1) - 40), f'{text}', font=myFontsmall,
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

def parse_pickles():
    evaluation_files = glob.glob('evaluationdata/*.pkl')
    data = []
    for file_name in evaluation_files:
        with open(file_name, 'rb') as file:
            data.append(pickle.load(file))
    df = pd.DataFrame(data[0])
    for entry in data[1:]:
        df = df.append(entry, ignore_index = True)
    return df

def preference_bar_graph():
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
    bar_without = ax.bar(X + 0.00, data[:, 0], color=(0, 0, 1, 0.7), width=0.33, label="Without tool")
    bar_with = ax.bar(X + 0.33, data[:, 1], color=(0, 1, 0, 0.7), width=0.33, label="With tool")
    #ax.hlines(50, -1, 6, color=(1, 0, 0, 0.7))
    ax.set_ylabel('Percentage')
    ax.set_xticks(X + 0.33 / 2, labels)
    ax.legend()
    ax.bar_label(bar_without, padding=3)
    ax.bar_label(bar_with, padding=3)
    #fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    #randomize evaluation set
    #evaluation_set = constants.evaluation_set[1:]
    #random.shuffle(evaluation_set)
    #constants.evaluation_set[1:] = evaluation_set

    # preference_bar_graph()

    #start tool
    data_frame = parse_pickles()
    win = visualization_tool.Tool(analysis_data=data_frame)
    win.showMaximized()
    # win.box_plot_images()
    # win.save_images((0, 0))
    avp = win.available_datasets_projections()
    save_sphere_and_histograms(avp)
    pg.exec()

