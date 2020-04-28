import cv2 as cv
import numpy as np
from helper import showImage

class SudokuImageProcessor(object):
    def __init__(self, img_path):
        self.img_path = img_path

    def getDist(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def preProcessImage(self, img, dilate=False):
        if img is None:
            return None
        img = cv.GaussianBlur(img, (9, 9), 0)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        img = cv.bitwise_not(img)

        if dilate:
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
            img = cv.dilate(img, kernel)

        return img

    def playHough(self, img):
        copy = np.copy(img)
        copy = self.preProcessImage(copy, False)
        side = img.shape[0]
        line_img = 255 * np.ones_like(img, dtype=np.uint8)
        lines = cv.HoughLinesP(copy, 1, np.pi/180, 100, 100, 10)
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if self.getDist((x1, y1), (x2, y2)) >= side / 9:
                print((x1, y1), (x2, y2))
                cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

        for x in range(side):
            for y in range(side):
                if line_img.item(y, x) == 0:
                    img[x][y] = 255

        return img

    def getBoardContours(self, img):
        ext_contours, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        polygon = None
        for contour in ext_contours:
            if polygon is None or cv.contourArea(contour) > cv.contourArea(polygon):
                polygon = contour

        vals = [0, float('inf'), float('inf'), 0]
        # Bottom right, Top left, Bottom left, Top right
        points = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for point in (point_arr[0] for point_arr in polygon):
            x, y = point
            if x + y > vals[0]:
                vals[0], points[0] = x + y, point
            if x + y < vals[1]:
                vals[1], points[1] = x + y, point
            if x - y < vals[2]:
                vals[2], points[2] = x - y, point
            if x - y > vals[3]:
                vals[3], points[3] = x - y, point

        return points

    def cropAndWarp(self, img, points):
        bottom_right, top_left, bottom_left, top_right = points

        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        max_dist = max(self.getDist(p1, p2) for p1, p2 in zip(src, np.vstack((src[1:], src[0]))))

        dst = np.array([[0, 0], [max_dist - 1, 0], [max_dist - 1, max_dist - 1], [0, max_dist - 1]], dtype='float32')

        matrix = cv.getPerspectiveTransform(src, dst)

        img = cv.warpPerspective(img, matrix, (int(max_dist), int(max_dist)))

        if max_dist > 675:
            img = cv.resize(img, (675, 675))

        return img

    def getLargestFeature(self, img):
        h, w = img.shape[:2]

        centre = (w / 2, h / 2)
        dist_to_centre = float('inf')
        seed = None
        total_area = h * w

        copy = img.copy()
        for x in range(w):
            for y in range(h):
                if img[y][x] == 255 and self.getDist(centre, (x, y)) < dist_to_centre:
                    area = cv.floodFill(copy, None, (x, y), 64)[0]
                    if area >= total_area / 30:
                        seed = (x, y)
                        dist_to_centre = self.getDist(seed, centre)

        found_num = seed and dist_to_centre < ((w + h) / 10)

        if found_num:
            area = cv.floodFill(img, None, seed, 64)[0]

        x1, y1, x2, y2 = w, h, 0, 0
        for x in range(w):
            for y in range(h):
                pixel = img.item(y, x)
                if pixel != 64:
                    cv.floodFill(img, None, (x, y), 0)
                else:
                    x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x), max(y2, y)

        if found_num:
            area = cv.floodFill(img, None, seed, 255)[0]

        bounding_width = x2 - x1
        bounding_height = y2 - y1

        found_bounding = found_num and bounding_height > bounding_width

        return np.array([[x1, y1], [x2, y2]], dtype='float32') if found_bounding else None

    def getRectSnippetFromImg(self, img, rect):
        return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

    def getDigitBox(self, img):
        return self.getLargestFeature(img)

    def getDigits(self, img, show_nums=False):
        # Assumes img is a square
        edge = img.shape[0]
        cell_side = edge / 9
        res = {}

        copy = img.copy()
        copy = self.preProcessImage(copy, True)

        for x in range(9):
            for y in range(9):
                start = [x * cell_side, y * cell_side]
                end = [start[0] + cell_side, start[1] + cell_side]
                cell = self.getRectSnippetFromImg(copy, [start, end])
                digit_box = self.getDigitBox(cell)

                if digit_box is not None:
                    orig_cell = self.getRectSnippetFromImg(img, [start, end])
                    res[x, y] = self.getRectSnippetFromImg(orig_cell, digit_box)
                    if show_nums:
                        showImage(res[x, y])
                    res[x, y] = self.repositionDigit(res[x, y])
        return res

    def repositionDigit(self, digit, side=28):
        copy = digit.copy()
        
        copy = cv.GaussianBlur(copy, (3, 3), 0)
        # Also considered using a binary + otsu method for thresholding
        # copy = cv.threshold(copy, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

        copy = cv.adaptiveThreshold(copy, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 0)
        copy = cv.bitwise_not(copy)

        final = np.zeros((side, side), np.uint8)
        copy = cv.resize(copy, (6 if copy.shape[1] <= 14 else 16, 16))
        h, w = copy.shape[:2]

        ax, ay = (side - w) // 2, (side - h) // 2

        final[ay : ay + h, ax : ax + w] = copy

        final = final.astype('float32')
        final /= 255
        final = final.reshape(1, 28, 28, 1)

        return final

    # Used to help visualize the imaginary grid lines
    def placeGrid(self, img, row=9, col=9, color=255):
        h, w = img.shape[:2]
        cell_height, cell_width = h / row, w / row

        for r in range(row):
            x1, x2 = 0, w
            y = int(r * cell_height)
            cv.line(img, (x1, y), (x2, y), color, 1)

        for c in range(col):
            y1, y2 = 0, h
            x = int(c * cell_width)
            cv.line(img, (x, y1), (x, y2), color, 1)


    def getBoardNumbers(self):
        gray_img = cv.imread(self.img_path, cv.IMREAD_GRAYSCALE)
        
        gray_img_copy = np.copy(gray_img)
        img = self.preProcessImage(gray_img_copy, True)
        corners = self.getBoardContours(img)
        img = self.cropAndWarp(gray_img_copy, corners)

        return self.getDigits(img, False)