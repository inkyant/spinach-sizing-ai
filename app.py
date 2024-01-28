from shiny import *
from shiny.types import FileInfo, ImgData
from shinywidgets import output_widget, render_widget
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist
import plotly.graph_objs as go
import plotly.express as px
import datetime
import calendar
from growthfunc import growthfunc

from segment_image import segment_plant_image

# rsconnect deploy shiny "/Users/mtwatson/Desktop/hackathon app" --name mtwatson --title sizing

def grade(x):
    gradeCutoffsg = [3, 5]
    if(x <= gradeCutoffsg[0]):
        return("small")
    if(x < gradeCutoffsg[1]):
        return("baby")
    if(x >= gradeCutoffsg[1]):
        return("bunching")

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

def areaToMass(area):
    leafThicknesscm = 0.05 # https://onlinelibrary.wiley.com/doi/full/10.1002/jsfa.5780
    leafDensitygcm3 = 1
    return area * leafThicknesscm * leafDensitygcm3

def segmentImage(image):
    knownQuarterAreacm2 = 4.62244 # https://en.wikipedia.org/wiki/Quarter_(United_States_coin)
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
    contours, _ = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    masses = []
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
            leaf = cv2.resize(leaf, (224,224))
            meanColor = np.round(meanNonBlackColor(leaf)).astype(int)

            area = cv2.contourArea(contour)

            # filter leaves based on CNN classifier - code below is to train it
            # import datetime
            # cv2.imwrite("/Users/mtwatson/Desktop/training images/" + str(datetime.datetime.now()) + ".jpg", leaf)
            
            sizecm2 = area * (knownQuarterAreacm2 / knownQuarterAreaPixels) 
            massg = areaToMass(sizecm2)
            leaves.append(leaf)
            masses.append(massg)

    return masses, cv2.resize(thres, None, fx = 0.25, fy = 0.25) # cv2.vconcat(leaves)

def imshow(image):
    dir = Path(__file__).resolve().parent
    cv2.imwrite(str(dir / 'buffer.jpg'), image)
    imgBuffer: ImgData = {"src": str(dir / 'buffer.jpg')}
    return imgBuffer


#####################
# ui and server code --------------------------------------------------------------------------------------------------------
#####################

image_types = ["Leaves Image", "Plant Image"]

app_ui = ui.page_fluid(
    ui.column(12,
        ui.panel_title("PlantPredict"),
        ui.input_select("image_type", "Select Image Type", image_types),
        ui.input_file("imageFile", "Take image", accept=["image/*"], multiple=False, capture='environment'),
        # output_widget("distributionBoxplot"),
        output_widget("projectionPlot"),
        # ui.output_image("img"),
        align="center",
    )
)

def server(input, output, session):

    masses = reactive.Value(True)
    segmentations = reactive.Value(True)

    @reactive.Calc
    def parsed_file():
        file: list[FileInfo] | None = input.imageFile()
        if file is None:
            return None
        return file[0]["datapath"]

    @reactive.Effect
    @reactive.event(input.imageFile)
    def _():
        if input.image_type() == image_types[0]:
            theseMasses, theseSegmentations = segmentImage(cv2.imread(parsed_file()))
            segmentations.set(theseSegmentations)
        else:
            theseMasses = list(areaToMass(np.array(segment_plant_image(parsed_file()))))
        masses.set(theseMasses)
    
    # @output
    # @render_widget
    # def distributionBoxplot():
    #     if(masses() == True):
    #         return px.box(y = [0], labels = dict(y = "Individual leaf mass (g)"))
    #     df = px.data.tips()
    #     fig = px.box(y = masses(), points="all", labels=dict(y = "Individual leaf mass (g)"))
    #     return fig
    
    @output
    @render_widget
    def projectionPlot():
        base = datetime.datetime.today()
        dates = [base + datetime.timedelta(days = x) for x in range(14)]

        if (masses() == True):
            return

        theseMasses = masses()

        distributions = pd.DataFrame(columns=['Date', 'Baby Mass'])

        for date in dates:
            for i, mass in enumerate(theseMasses):
                if grade(mass) == "baby":
                    distributions.loc[len(distributions.index)] = [date, mass]
                theseMasses[i] = growthfunc(mass, calendar.month_name[date.month])

        if distributions.empty:
            print("distribution is empty")
            return

        # to find average
        # averages = distributions.groupby(distributions['Date'].dt.date).sum()
        
        # or instead find sum:
        # have to mess around with the Date becomeing the index due
        # to sum function not liking date objects.
        averages = distributions.groupby('Date').sum()
        averages.insert(1, "Date", [x for x in averages.index])
        averages.index = (a for a in range(averages.shape[0]))

        # scale axis of the mass.
        averages.loc[:, "Baby Mass"] = averages.loc[:, "Baby Mass"] / 5

        # print(np.array([grade(i) for i in distributions[['Mass']]]))

        fig1 = px.line(averages, x="Date", y="Baby Mass", labels = dict(x = "Date", y = "Baby leaf yield (g)"))
        fig1.update_traces(line=dict(color = 'rgba(9, 232, 69, 1)', width=8))

        fig2 = px.scatter(distributions, x="Date", y="Baby Mass", labels = dict(x = "Date", y = "Baby leaf yield (g)"))

        fig3 = go.Figure(data=fig1.data + fig2.data)

        return fig3

    @output
    @render.image
    def img():
        return imshow(segmentations())

app = App(app_ui, server)
app.run()