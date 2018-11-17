import glob
import os
from PIL import Image

stride=14
f_sub=33
f1,f2,f3 = 9,1,5
dir_name='sub_images'

files = glob.glob("images/low/*")
count=len(files)
print(count,"images found.")
os.makedirs(dir_name+'/low',exist_ok=True)
os.makedirs(dir_name+'/high',exist_ok=True)
pad=(f1+f2+f3-3)/2
for file in files:
    id=os.path.splitext(os.path.basename(file))[0]
    low=Image.open(file)
    high=Image.open(file.replace('low\\','high\\'))
    w,h = low.size
    for wi in range((w-f_sub)//stride+1):
        for hi in range((h-f_sub)//stride+1):
            box=(wi*14,hi*14,wi*14+33,hi*14+33)
            name=id+'_'+str(wi)+'_'+str(hi)+'.png'
            high.crop(box=box).save(dir_name+'/high/'+name)
            box=(wi*14+pad,hi*14+pad,wi*14+33-pad,hi*14+33-pad)
            low.crop(box=box).save(dir_name+'/low/'+name)
