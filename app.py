from shiny import *
from shiny.types import FileInfo
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from itertools import compress
from scipy.stats import mode
from scipy.spatial import cKDTree, distance_matrix
from scipy.spatial.distance import cdist
import plotly.graph_objs as go
import plotly.express as px
from shinywidgets import output_widget, register_widget

# rsconnect deploy shiny /Users/mtwatson/Desktop/hackathon/app --name mtwatson --title leafSizing

def cropSquareFromContour(c, img):
    rect = cv2.minAreaRect(c)
    rect = list(rect)
    rect[1] = (max(rect[1]), max(rect[1]))
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(max(rect[1]))
    height = int(max(rect[1]))
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def meanNonBlackColor(image):
    indices = np.array([np.any(pixel != [0, 0, 0], axis=-1) for pixel in image])
    return np.mean(np.array(image[indices]), axis=0)

def segmentImage(image):
    knownQuarterAreacm2 = 4.62244
    thres = image
    thres = cv2.resize(image, None, fx = 0.25, fy = 0.25)
    thres = cv2.blur(thres, (10, 10))  
    thres = cv2.erode(thres, (50, 50)) 
    greenDistances = cdist(thres.reshape(-1, 3), [[-10,255,-10]])
    thres = greenDistances.reshape(thres.shape[0], thres.shape[1], 1).astype("uint8")
    thres = cv2.adaptiveThreshold(thres,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 2501, 7)
    kernel = np.ones((9, 9), np.uint8)
    thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
    thres = cv2.resize(thres, None, fx = 4, fy = 4)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    knownQuarterAreaPixels = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > (thres.shape[0] * thres.shape[1] * 0.0001) and area < (thres.shape[0] * thres.shape[1] * 0.1)):
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * (area/(perimeter * perimeter))
            if(circularity > 0.65):
                canvas = np.zeros(thres.shape).astype("uint8")
                canvas = cv2.drawContours(canvas, [contour], -1, 255, cv2.FILLED)
                leafMask = cropSquareFromContour(contour, canvas)
                leaf = cropSquareFromContour(contour, image)
                leaf[(255 - leafMask).astype("bool"), :] = [0,0,0]
                leaf[(255 - cropSquareFromContour(contour, thres)).astype("bool"), :] = [0,0,0]
                leaf = cv2.resize(leaf, (100,100))
                knownQuarterAreaPixels = area

    leaves = []
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        rectArea = rect[1][0] * rect[1][1]
        if(rectArea > (thres.shape[0] * thres.shape[1] * 0.01) and rectArea < (thres.shape[0] * thres.shape[1] * 0.5)):
            canvas = np.zeros(thres.shape).astype("uint8")
            canvas = cv2.drawContours(canvas, [contour], -1, 255, cv2.FILLED)
            leafMask = cropSquareFromContour(contour, canvas)
            leaf = cropSquareFromContour(contour, image)
            leaf[(255 - leafMask).astype("bool"), :] = [0,0,0]
            leaf[(255 - cropSquareFromContour(contour, thres)).astype("bool"), :] = [0,0,0]
            leaf = cv2.resize(leaf, (100,100))
            meanColor = np.round(meanNonBlackColor(leaf)).astype(int)

            area = cv2.contourArea(contour)
            # if(np.mean(meanColor - [40, 70, 60])) < 15:
            leafSizecm2 = area * (knownQuarterAreacm2 / knownQuarterAreaPixels)
            leaves.append(leaf)
                
    return(cv2.vconcat(leaves))

def imshow(image):
    dir = Path(__file__).resolve().parent
    cv2.imwrite(str(dir / 'buffer.jpg'), image)
    imgBuffer: ImgData = {"src": str(dir / 'buffer.jpg')}
    return imgBuffer

app_ui = ui.page_fluid(
    ui.input_file("imageFile", "Take image", accept=["image/*"], multiple=False, capture='environment'),
    ui.output_image("img")
)

def server(input, output, session):
    print(imshow)
    @reactive.Calc
    def parsed_file():
        file: list[FileInfo] | None = input.imageFile()
        if file is None:
            return None
        return cv2.imread(
            file[0]["datapath"]
        )

    @output
    @render.image
    def img():
        image = parsed_file()
        image = segmentImage(image)

        return(imshow(image))

app = App(app_ui, server)
app.run()