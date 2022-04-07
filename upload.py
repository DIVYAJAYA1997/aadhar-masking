# jayadev@geojit.com on 17 Feb 2022
# install tesseract-ocr before pip install pytesseract
#from google.colab.patches import cv2_imshow

#### REQUIREMENTS #####
#!pip install pytesseract
#!pip install opencv-contrib-python
#!pip install matplotlib
#!pip install flask

#!apt install tesseract-ocr
#### REQUIREMENTS #####


# jayadev@geojit.com on 17 Feb 2022
# install tesseract-ocr before pip install pytesseract
#from google.colab.patches import cv2_imshow

#### REQUIREMENTS #####
#!pip install pytesseract
#!pip install opencv-contrib-python
#!pip install matplotlib
#!pip install flask

#!apt install tesseract-ocr
#### REQUIREMENTS #####cd cd

from pyexpat import model
from flask import Flask, render_template, request, send_from_directory, make_response,url_for
import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
import os

##### CHANGE HERE - set tesseract-ocr binary folder #######
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'/content'

app = Flask(__name__)

def mask_image(filename):
    img = cv2.imread(filename)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng', config=tessdata_dir_config)#to localize each area of text int the input img
    n_boxes = len(d['level'])

    overlay = img.copy()

    text1 = ""
    text2 = ""
    for i in range(n_boxes):#creating rectangle box for masking
        text = d['text'][i]
        # FUZZY LOGIC == Aadhaar number is a 12 digit number grouped - 4 digits * 3 ex: 1111 2222 3333
        # detect if first two words are numeric and check if they have 4 digits in them
        # mask first two 4 digits
        #print(text, end = " ")
        if (text == "DEMAT"):
            print("DEMAT PAGE")
        #else:
            #print("NOT A DEMAT PAGE")
        if (d['text'][i].isnumeric() and len(d['text'][i]) == 4 and d['text'][i+1].isnumeric() and len(d['text'][i+1]) == 4):
            text1 = d['text'][i] #First 3 digits of Aadhaar
            text2 = d['text'][i+1] #Second 3 digits of Aadhaar
            text3 = d['text'][i+2] #Third 3 digits of Aadhaar
            #print(d['text'][i])
            if len(text1+text2+text3) == 12:
                a_num = text3
                print(a_num)
            
            if text == text1 or text == text2:
                #print('Masking Text ' + text, flush=True)
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                (x1, y1, w1, h1) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                (x2, y2, w2, h2) = (d['left'][i + 2], d['top'][i + 2], d['width'][i + 2], d['height'][i + 2])
                cv2.rectangle(img, (x, y), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                cv2.rectangle(overlay, (x, y), (x1 + w1, y1 + h1), (0, 165, 255), -1)
                #cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                #cv2.rectangle(overlay, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), -1) 
                
            
    print("----")
    #cv2_imshow(img)

    alpha = 1  # Transparency factor for the mask.
    # Following line overlays transparent rectangle over the image
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    r = 1000.0 / img_new.shape[1]  # resizing image without loosing aspect ratio
    dim = (1000, int(img_new.shape[0] * r))
    # perform the actual resizing of the image and show it
    resized = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
    #cv2_imshow(img_new)
    #cv2.imshow("Image", img_new)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()
    cv2.imwrite(filename, img_new)
    return a_num

@app.route('/mask-text', methods = ['POST'])
def mask_text():
   print("valid")
   if request.method == 'POST':
       f = request.files['file']
       f.save(f.filename)
       
        # Failure to return a redirect or render_template
        

       a_num = mask_image(f.filename)
      #print(os.path.join(app.root_path))
       response = make_response(send_from_directory(os.path.join(app.root_path), 
                                f.filename, mimetype = 'image/png'))
      ## Send detected last 4 digits in the HTTP header
       response.headers['last4digits'] = a_num
       return response

    #  return send_from_directory(
    #   os.path.join(app.root_path),
    #   f.filename,
    #   mimetype = 'image/png') 
    
#if __name__ == '__main__':
#   app.run(debug = True)

if __name__ == '__main__':
    app.run(debug = True)



