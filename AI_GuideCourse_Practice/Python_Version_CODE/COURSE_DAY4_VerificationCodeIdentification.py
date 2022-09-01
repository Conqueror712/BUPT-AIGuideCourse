# 验证码识别
import pytesseract
from PIL import Image

# 1. 引入 Tesseract 程序
pytesseract.pytesseract.tesseract_cmd = r'D:\Coding\Python\Pytesseract\Tesseract-OCR\tesseract.exe'
# 2. 使用 Image 模块下的 Open()函数打开图片
image = Image.open('D:\\BUPT\\基础课程\\人工智能导论实践\\Data_Set\\3-2\\vcode1.gif', mode='r')
print(image)
# 3. 识别图片文字
code = pytesseract.image_to_string(image)
print(code)

