import cv2 as cv
from PIL import Image
import os
import time
import numpy

#pt.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

src_path = 'DDD/'


def main():
    seq_list = create_all_sequences()
    for seq in seq_list:
        array = numpy.asarray(seq)
        sub1 = bg_subtraction(seq)
        sub2 = bg_subtraction(list(reversed(array)))
        i = 0
        for s in sub1:
            array = numpy.asarray(sub2)
            rev = list(reversed(array))
            res = cv.multiply(s,rev[i])
            compare_results(seq[i],res)
            i += 1


def create_all_sequences():
    all_images = os.listdir(src_path)
    rest_img = all_images
    all_seq = []
    while(True):
        seq, rest_img = create_sequence(rest_img)
        all_seq.append(seq)
        if len(rest_img)== 0:
            break
    return all_seq


def create_sequence(images):
    seq = [images[0]]
    seq_path = src_path + seq[0]
    seq_time_str = Image.open(seq_path)._getexif()[36867]
    seq_time = time.strptime(seq_time_str, "%Y:%m:%d %H:%M:%S")

    for img in images:
        img_path = src_path + img
        img_time_str = Image.open(img_path)._getexif()[36867]
        img_time = time.strptime(img_time_str, "%Y:%m:%d %H:%M:%S")
        t1 = time.mktime(img_time)
        t2 = time.mktime(seq_time)
        if abs(t1 - t2) < 200 and img != images[0]:
            seq.append(img)

    for s in seq:
        images.remove(s)
    return seq, images


def bg_subtraction(seq):
    fgbg = cv.createBackgroundSubtractorMOG2(len(seq),32)
    kernels = [numpy.ones((7,7), numpy.uint8) ,numpy.ones((5,5), numpy.uint8),numpy.ones((3,3), numpy.uint8)]
    bg_subtracted = []
    old_close = None
    for img_str in seq:
        img = cv.imread(src_path+img_str,0)
        blur = cv.GaussianBlur(img, (5, 5), 0)
        equ = cv.equalizeHist(blur)
        fg = fgbg.apply(equ)
        for k  in kernels:
            open = cv.morphologyEx(fg, cv.MORPH_OPEN, k)
            close = cv.morphologyEx(open, cv.MORPH_CLOSE, k)
            if old_close is None:
                old_close = close
            if numpy.average(close) > 1.25:
                break

        if numpy.average(close) < 0.7:
            close = old_close

        old_close = close
        res = color_filter(img_str,close)
        bg_subtracted.append(res)
    return bg_subtracted


def color_filter(img_str, closure):
    colored = cv.imread(src_path + img_str)
    hsv = cv.cvtColor(colored, cv.COLOR_BGR2HSV)
    green_mask = cv.inRange(hsv, (36, 0, 0), (70, 255, 255))
    imask = green_mask > 0
    green_filtered = numpy.zeros_like(colored, numpy.uint8)
    green_filtered[imask] = colored[imask]
    temp_bgr = cv.cvtColor(green_filtered, cv.COLOR_HSV2BGR)
    green_to_gray = cv.cvtColor(temp_bgr, cv.COLOR_BGR2GRAY)
    k, green_thresh = cv.threshold(green_to_gray, 2, 255, cv.THRESH_BINARY_INV)
    p, res = cv.threshold(closure, 20, 255, cv.THRESH_BINARY)
    result = cv.multiply(green_thresh, res)
    return result


def compare_results(img_str, detected):
    img = cv.imread(src_path + img_str)
    red_det = cv.cvtColor(detected, cv.COLOR_GRAY2BGR)
    img_rgb = numpy.zeros_like(img, numpy.uint8)
    img_rgb[:,:,2] = 255
    red_thresh = cv.multiply(red_det,img_rgb)
    comparison = cv.addWeighted(img,0.7,red_thresh,0.3,0)
    show_image(comparison)


def show_image(img):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 600, 600)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


