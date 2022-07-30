import random
import time


import constants
import visualization_tool
import pyqtgraph as pg
import glob
import pickle
import pandas as pd
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
    projections = list(avp[list(avp.keys())[0]])
    projections.sort()
    for projection in projections:
        datasets = list(avp.keys())
        datasets.sort()
        for index, dataset in enumerate(datasets):
            images = []
            for metric in constants.metrics:
                if index == len(datasets) - 1:
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
            images.append(Image.new('RGB', (images[0].width * 3 + 40, images[0].height), (255, 255, 255)))
            images = [Image.new('RGB', (images[0].width * 2, images[0].height), (255, 255, 255))] + images
            image = get_concat_h(images[0], images[1])
            for i in range(2, len(images)):
                image = get_concat_h(image, images[i])

            I1 = ImageDraw.Draw(image)
            str = dataset
            if dataset == 'WisconsinBreastCancer':
               str = 'WBC'
            I1.text((sum([im.width for im in images[:len(images) - 1]]) + 40, 50), f'{str}', font=myFont,
                    fill=(0, 0, 0))
            if index == 0:
                I1.text((10, 20), projection, font=myFont, fill=(0, 0, 0))
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

if __name__ == '__main__':
    #randomize evaluation set
    evaluation_set = constants.evaluation_set[1:]
    random.shuffle(evaluation_set)
    constants.evaluation_set[1:] = evaluation_set

    #start tool
    data_frame = parse_pickles()
    win = visualization_tool.Tool(analysis_data=data_frame)
    win.showMaximized()
    #win.box_plot_images()
    avp = win.available_datasets_projections()
    save_sphere_and_histograms(avp)
    # win.save_images((0, 0))
    pg.exec()

