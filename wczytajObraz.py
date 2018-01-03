import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
# import resizeImg
from featureMatchingHomogr import featureMatchingHomogr
from matplotlib import pyplot as plt

root = Tk()
root.fileName = filedialog.askopenfile(filetypes=(("png files", "*.png"), ("All files", "*.*")))
logo = ""
dyst = 0.7
font = cv2.FONT_HERSHEY_SIMPLEX
try:
    img2 = cv2.imread(root.fileName.name, 0)
    root.destroy()
    img22 = img2
    img3 = []
    MIN_MATCH_COUNT = 5
    mmin = 0
    while len(img3) == 0:
        for num in range(13):
            img1 = cv2.imread(r'MercedesLogo\m' + str(num+1) + '.png', 0)
            img11 = cv2.imread(r'MercedesLogo\m' + str(num+1) + '.png', 0)

            img33 = []
            if mmin < MIN_MATCH_COUNT:
                mmin = MIN_MATCH_COUNT

            img33, MIN_MATCH_COUNT = featureMatchingHomogr(img1, img11, img2, img22, MIN_MATCH_COUNT, img3, dyst)

            if MIN_MATCH_COUNT > mmin:
                img3 = []
                img3 = img33
                logo = "Mercedes"

            # w, h, ssl = img3.shape


        for num in range(15):
            img1 = cv2.imread(r'ToyotaLogo\t' + str(num+1) + '.png', 0)
            img11 = cv2.imread(r'ToyotaLogo\t' + str(num+1) + '.png', 0)

            img33 = []
            if mmin < MIN_MATCH_COUNT:
                mmin = MIN_MATCH_COUNT

            print("Toyota:")

            img33, MIN_MATCH_COUNT = featureMatchingHomogr(img1, img11, img2, img22, MIN_MATCH_COUNT, img3, dyst)

            if MIN_MATCH_COUNT > mmin:
                img3 = []
                img3 = img33
                logo = "Toyota"
        dyst = dyst + 0.1
except ValueError:
    print("Coś nie pykło")



cv2.putText(img2, logo, (0, 100), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
plt.imshow(img3, 'gray'), plt.show()
# plt.imshow(img2, 'gray'), plt.show()

print("Nic nie znaleziono.")
