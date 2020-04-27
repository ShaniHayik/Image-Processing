"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import argparse
from tkinter import Tk, Canvas

import click as click
import cv
import cv2
import nothing as nothing
import numpy as np
import ex1_utils

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

"""
  GUI for gamma correction
  :param img_path: Path to the image
  :param rep: grayscale(1) or RGB(2)
  :return: None
  """

def gammaDisplay(img_path: str, rep: int):
    running = True
    img = ex1_utils.imReadAndConvert(img_path, rep)
    alpha_slider = 0
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp = np.copy(img)

    def on_trackbar(val):
        gamma = float(val)
        gamma = gamma/100
        img = np.uint8(255 * np.power(temp, gamma))  #calculate new img
        cv2.imshow('nameWin', img)

    cv2.namedWindow('nameWin')
    cv2.createTrackbar('Gamma', 'nameWin', alpha_slider, 200, on_trackbar)  #making trackbar

    while running == True:
        cv2.waitKey()
        root = Tk()
        root.protocol("WM_DELETE_WINDOW", close_window)  #when closing the window
        cv = Canvas(root, width=200, height=200);
        cv.pack()


def close_window(): #when closing the window
  global running
  running = False


def main():
    gammaDisplay('bac_con.png', 1)


if __name__ == '__main__':
    main()
