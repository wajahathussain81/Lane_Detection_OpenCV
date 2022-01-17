import cv2 as cv
import numpy as np
import logging


def __init__(self, car=None):
    logging.info('Creating a HandCodedLaneFollower...')
    self.car = car
    self.curr_steering_angle = 90


def detect_edges(frame):
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(gray_img, 150, 255)
    # cv.imshow('mask', mask)

    edges = cv.Canny(mask, 200, 400)
    # cv.imshow('edges', edges)
    return edges


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height ),
        (width * 1/2, height * 7/10),
        (width * 18/20 , height),
    ]], np.int32)

    cv.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    # cv.imshow('cropped edges', cropped_edges)
    return cropped_edges


def detect_line_segments(cropped_edges):
    rho = 1
    angle = np.pi / 180
    min_threshold = 10
    line_segments = cv.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                   np.array([]), minLineLength=100, maxLineGap=50)

    return line_segments


def average_slope_intercept(frame, line_segments):
    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    for line_segment in line_segments:
        x1, y1, x2, y2 = line_segment.reshape(4)
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        intercept = fit[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)

    left_line = make_points(frame, left_fit_average)

    right_fit_average = np.average(right_fit, axis=0)
    right_line = make_points(frame, right_fit_average)

    # logging.debug('lane lines: %s' % lane_lines)

    return np.array([left_line, right_line])


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 -200)

    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=5):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    # cv.imshow('line image', line_image)
    return line_image


cap = cv.VideoCapture('Test.mp4')

while True:
    ret, frame = cap.read()
    # cv.imshow('frame', frame)
    detect_edges(frame)
    region_of_interest(detect_edges(frame))
    display_lines(frame, average_slope_intercept(frame, detect_line_segments(region_of_interest(detect_edges(frame)))))
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

frame.release()
cv.destroyAllWindows()


