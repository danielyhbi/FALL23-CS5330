# In class exercise

import sys
import os
import numpy as np
import cv2

def main(argv):
    
    orgrows = 0
    orgcolors = 0
    
    srcdir = argv[1]
    
    filelist = os.listdir( srcdir)
    
    buildmtc = True
    
    for filename in filelist:
        
        print("Procecssing file %s" % filename)
        
        suffix = filename.split(".")[-1]
        
        if not ('pgm' in suffix):
            continue
        
        src = cv2.imread(srcdir+"/"+filename)
        
        # make the image a single channel and resize the image
        src = [:, :, 1]
        src=cv2. resize(src, (src.shape[1]*4, src.shape[0]*4), interpolation=cv2.INTER_AREA)
        
        # resize the image to a single vector
        newImg = np.reshape(src, src.shape[0]*src.shape[1])
        
        if buildmtx:
            amtx = newImage.astype("float32")
            buildmtx = False
            