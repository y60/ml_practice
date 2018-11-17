import glob
import os
from PIL import Image
files=glob.glob('high/*.png')
print(len(files),'images found.')
for file in files:
    im=Image.open(file)
    w,h = im.size
    low=im.resize((w//2,h//2)).resize((w,h))
    low.save(file.replace('high\\','low\\'))
    break
    