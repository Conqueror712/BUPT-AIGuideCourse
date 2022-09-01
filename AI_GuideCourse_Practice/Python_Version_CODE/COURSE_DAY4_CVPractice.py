from PIL import Image

im = Image.open("D:\\Coding\\Git\\AI_Course\\mmediting\\tests\\data\\merged\\GT05.jpg")  # 图像的绝对路径
im_L = im.convert("L")  # convert函数将原始图片（例如png）转化为对应的像素值，再将像素值转化成tensor之后进行模型的训练

'''
convert中可设置转换模式，介绍比较常用的三种模式：
（1）RGB模式
（2）1模式  转化为为二值图像，非黑即白，每个像素用8个bit表示，0表示黑，255表示白。
（3）L模式  转化为为灰色图像，每个像素用8个bit表示，0表示黑，255表示白，0~255代表不同的灰度。需要注意的是，在PIL中，RGB是通过以下公式转化为L的：
                                        L = R * 299/1000 + G * 587/1000 + B * 114/1000
'''

box = (560, 1000, 1800, 1800)
region = im_L.crop(box) # crop进行图像裁剪
region.save("D:\\Coding\\Git\\AI_Course\\mmediting\\tests\\data\\merged\\NEW_GT05.jpg")
region.show()


im1 = Image.open("d:\\1.jpg")
im2 = Image.open("d:\\2.jpg")

'''
Image.split()方法用于将图像分成单独的波段。此方法从图像返回单个图像带的元组。
分割“RGB”图像会创建三个新图像，每个图像都包含一个原始波段(红色，绿色，蓝色)的副本。
'''

r1, g1, b1 = im1.split()
r2, g2, b2 = im2.split()
print(r1.mode, r1.size, g1.mode, g1.size)
print(r2.mode, r2.size, g2.mode, g2.size)
new_im = [r1, g2, b2]
print(len(new_im))
im_merge = Image.merge("RGB", new_im)
im_merge.show()