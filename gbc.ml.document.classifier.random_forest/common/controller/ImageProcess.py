from random import randrange

import cv2
import imutils
import numpy as np
from scipy.signal import convolve2d


class ImageProcess:

    @staticmethod
    def load_image(file):
        img = cv2.imread(file, 1)
        if img is None:
            raise FileNotFoundError
        # show_img(img, 'loaded')
        return img
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def crop_image(file):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        crop_img = img[0:200, 0:200]  # y,x
        # show_img(crop_img, 'cropped')
        return crop_img

    @staticmethod
    def crop_image_loaded(img, y, x):
        crop_img = img[0:y, 0:x]  # y,x
        # show_img(crop_img, 'cropped')
        return crop_img

    @staticmethod
    def resize_image(file):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        new_img = cv2.resize(img, (600, 800))  # w,h
        # show_img(new_img, 'resized')
        return new_img

    @staticmethod
    def resize_image_loaded(img, width, height):
        new_img = cv2.resize(img, (width, height))
        # show_img(new_img, 'resized')
        return new_img

    @staticmethod
    def resize_factor_image_loaded(img, width_factor, height_factor):
        new_img = cv2.resize(img, None, width_factor, height_factor)
        # show_img(new_img, 'resized')
        return new_img

    @staticmethod
    def show_img(img, text=''):
        cv2.imshow(text, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_img_np(self, data):
        img = (data * 255).astype(np.uint8)
        self.show_img(img)

    @staticmethod
    def rotate(img, angle=0):
        if angle == 0:
            angle = randrange(-3, 3)
            # print(angle)
        rotated = imutils.rotate(img, angle)
        # show_img(self, self, rotated, 'rotated')
        return rotated

    @staticmethod
    def translate(img, x=0, y=0):
        if y == 0:
            y = randrange(-5, 5)
            # print(y)
        if x == 0:
            x = randrange(-5, 5)
            # print(x)
        translated = imutils.translate(img, x, y)
        # show_img(translated, 'translated')
        return translated

    @staticmethod
    def zoom(img):
        new_img = img
        height, width, channels = img.shape
        up_down = randrange(-1, 2)
        if up_down == 1:
            new_img = cv2.pyrUp(img, dstsize=(width * 2, height * 2))
        elif up_down == 0:
            new_img = cv2.pyrDown(img, dstsize=(width // 2, height // 2))
        # show_img(new_img, 'zoomed')
        # print(new_img.shape)
        return new_img

    @staticmethod
    def brightness(img, value=0):
        if value == 0:
            value = randrange(0, 10)
            # print(value)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        new_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return new_img

    def generate_examples(self, img, num=10):
        examples = [self.invert(self.denoise(self.gray_image(img)))]
        for _ in range(num):
            example = self.translate(self.rotate(examples[0]))
            examples.append(example)

        return examples

    @staticmethod
    def invert(img):
        return cv2.bitwise_not(img)

    def kill_hermits2(self, img):
        result = img.copy()
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        mask = convolve2d(img, kernel, mode='same', fillvalue=1)
        # print(mask)
        result[np.logical_and(mask == 8, img == 0)] = 1
        self.show_img(result, 'cleaned')
        return result

    def kill_hermits3(self, img):
        result = img.copy()
        kernel = np.ones((4, 4))
        kernel[1, 1] = 0
        kernel[1, 2] = 0
        mask = convolve2d(img, kernel, mode='same', fillvalue=1)
        # print(mask)
        result[np.logical_and(mask == 12, img == 0)] = 1
        self.show_img(result, 'cleaned')
        return result

    @staticmethod
    def denoise(img):
        result = img.copy()
        cv2.fastNlMeansDenoising(img, result, 10.0, 7, 21)
        return result

    @staticmethod
    def no_mod(img):
        return img

    def kill_hermits(self, img):
        result = img.copy()
        height, width, channels = img.shape
        # print(img.shape)
        for y in range(height):
            up = y > 0
            down = y < height - 1
            for x in range(width):
                left = x > 0
                right = x < width - 1
                # print(y, x)
                if result[y, x][0] == 0:
                    has_neighbour = False
                    if up and result[y - 1, x][0] == 0.0:
                        has_neighbour = True
                    if down and result[y + 1, x][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if left and result[y, x - 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if right and result[y, x + 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if up and left and result[y - 1, x - 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if up and right and result[y - 1, x + 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if down and left and result[y + 1, x - 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if down and right and result[y + 1, x + 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True

                    if not has_neighbour:
                        result[y, x] = 255

        self.show_img(result, 'cleaned')
        return result

    @staticmethod
    def black_white_image(img):
        # (thresh, im_bw) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        im_bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
        return im_bw

    @staticmethod
    def black_white_image2(img, limit=150):
        result = img.copy()
        height, width = img.shape
        # print(img.shape)
        for y in range(height):
            for x in range(width):
                if result[y, x] > limit:
                    result[y, x] = 255
                else:
                    result[y, x] = 0
        return result

    @staticmethod
    def gray_image(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
