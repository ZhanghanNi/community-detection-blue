import pytesseract
from PIL import Image

img = Image.open("comm_ss.png")
text = pytesseract.image_to_string(img)
print(text)