# 画像を配列に格納して表示
import glob
from PIL import Image
import numpy

 
files = glob.glob('png/test/*.png')

SIZE = 28
for img in files:
    image = Image.open(img).convert('L')
    image = image.resize((SIZE, SIZE), Image.LANCZOS)
    img_gray_resize = numpy.array(image)
    
    for x in range(0, SIZE):
        for y in range(0, SIZE):
            print(img_gray_resize[x,y], end='')
        print('')
    print('')