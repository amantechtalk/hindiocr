import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from googletrans import Translator
trans = Translator()
import os
import tempfile
from pdf2image import convert_from_path
from PyPDF3 import PdfFileWriter ,PdfFileReader
import easyocr
reader = easyocr.Reader(['hi','en'])

path="C:/Users/amank/Downloads/"
output_folder_path="C:/Users/amank/Downloads/output/"
pdfname=path+"PDF3.pdf"
poppler_path = r'C:/Users/amank/Downloads/Release-22.04.0-0 (3)/poppler-22.04.0/Library/bin'
images = convert_from_path(pdfname,poppler_path=poppler_path)
i=1
len=len(images)
print("Number of pages in PDF="+str(len))
for image in images:
    image.save(output_folder_path +str(i)+'.jpg','JPEG')
    i=i+1
print(i)    






try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#read your file
for u in range(3,int(i)):
 file=r"C:/Users/amank/Downloads/output/"+str(u)+".jpg"
 img = cv2.imread(file,0)
 img.shape

 #thresholding the image to a binary image
 thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

 #inverting the image 
 img_bin = 255-img_bin
 cv2.imwrite('C:/Users/amank/Downloads/output/cv_inverted.jpg',img_bin)
 #Plotting the image to see the output
 #####plotting = plt.imshow(img_bin,cmap='gray')
 #####plt.show()

 # countcol(width) of kernel as 100th of total width
 kernel_len = np.array(img).shape[1]//70
 kernel_len1 = np.array(img).shape[1]//70
 # Defining a vertical kernel to detect all vertical lines of image 
 ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
 # Defining a horizontal kernel to detect all horizontal lines of image
 hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len1, 1))
 # A kernel of 2x2
 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
 
 #Use vertical kernel to detect and save the vertical lines in a jpg
 image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
 vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
 cv2.imwrite("C:/Users/amank/Downloads/output/vertical.jpg",vertical_lines)
 #Plot the generated image
 ########plotting = plt.imshow(image_1,cmap='gray')
 #########plt.show()    
 
 #Use horizontal kernel to detect and save the horizontal lines in a jpg
 image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
 horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
 cv2.imwrite("C:/Users/amank/Downloads/output/horizontal.jpg",horizontal_lines)
 #Plot the generated image
 #######plotting = plt.imshow(image_2,cmap='gray')
 ###plt.show()  

 # Combine horizontal and vertical lines in a new third image, with both having same weight.
 img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
 #Eroding and thesholding the image
 img_vh = cv2.erode(~img_vh, kernel, iterations=2)
 thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 cv2.imwrite("C:/Users/amank/Downloads/output/img_vh.jpg", img_vh)
 bitxor = cv2.bitwise_xor(img,img_vh)
 bitnot = cv2.bitwise_not(bitxor)
 #Plotting the generated image
 #####plotting = plt.imshow(bitnot,cmap='gray')
 #####plt.show()  

# Detect contours for following box detection
 contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)









 def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [list(cv2.boundingRect(c)) for c in cnts]
    boundingBoxes = list(boundingBoxes)
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    list(boundingBoxes)
    
    bitnot = cv2.bitwise_not(image_1)
    bita = cv2.bitwise_and(image_1,bitnot)
    boundingBoxes1=[]
    boundingBoxes2=[]
    cnts1=[]
    
    print(np.shape)
    for c in cnts:
        a=list(cv2.boundingRect(c))
        if a[3]>185 and a[3]<2055:
            cnts1.append(c)
    for  x in boundingBoxes:
        #print(cnts)
        if x[3]>185 and x[3]<205:
            boundingBoxes1.append(x)
          
            bitnot=cv2.rectangle(bita,x, (255, 255, 255),2)
            #print(x)
    
    for  x in boundingBoxes:
        if x[3]>135 and x[3]<145:
            boundingBoxes2.append(x)
            
            bitnot=cv2.rectangle(bita,x, (255, 255, 255),-1)
            #print(x)
    
    ###plotting = plt.imshow(bitnot,cmap='gray')
    ######plt.show()
            
    return (cnts1, boundingBoxes1,boundingBoxes2)

# Sort all the contours by top to bottom.
 contours,  boundingBoxes,boundingBoxes2 = sort_contours(contours, method="top-to-bottom")
 bitnot = cv2.bitwise_not(image_1)
 bita = cv2.bitwise_and(image_1,bitnot)
 for  x in boundingBoxes2:
    if x[3]>135 and x[3]<145:
    
            
        bitnot=cv2.rectangle(img,x, (255, 255, 255),-1)
        #print(x)
 for  x in boundingBoxes2:
    if x[3]>0 and x[3]<100:
    
            
        bitnot=cv2.rectangle(img,x, (255, 255, 255),-1)
        #print(x)        
 #plotting = plt.imshow(bitnot,cmap='gray')
 #plt.show()



 boundingBoxes = list(boundingBoxes)
 a=np.shape(boundingBoxes)
 print(a)
 print(boundingBoxes[1])
 b=int(a[0])
 print(a)
 print(b)
 #Creating a list of heights for all detected boxes
 for i in range (0,b) :
   heights=boundingBoxes[i][3]
 #Get mean of heights
 mean = np.mean(heights) 
 
 #Create list box to store all boxes in  
 box = []
 # Get position (x,y), width and height for every contour and show the contour on image
 for c in contours:
     x, y, w, h = cv2.boundingRect(c)
     if (w<1000 and h<500):
         image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
         box.append([x,y,w,h])
         
 ########plotting = plt.imshow(image,cmap='gray')
 ######plt.show() 

 #Creating two lists to define row and column in which cell is located
 row=[]
 column=[]
 j=0
 print("arham")
 print(box)
 #Sorting the boxes to their respective row and column
 for i in range(np.shape(box)[0]):    
         
     if(i==0):
         
         column.append(box[i])
         previous=box[i]    
     
     else:
         if(box[i][1]<=previous[1]+mean/2):
         
             column.append(box[i])
             previous=box[i]            
              
             if(i==np.shape(box)[0]-1):
                 
                 row.append(column)        
             
         else:
              
            
             row.append(column)
             column=[]
             previous = box[i]
             
             column.append(box[i])
 print(row)
 print(column)  

 #calculating maximum number of cells
 countcol = 0
 for i in range(np.shape(row)[0]):
     countcol = np.shape(row[i])[0]
     if countcol > countcol:
         countcol = countcol
 print(type(int(row[1][1][0])))
 print(type(row[1][1]))
 print(type(row[1]))
 #Retrieving the center of each column
 center = [(int(row[i][j][0])+int(row[i][j][2]))/2 for j in range(np.shape(row[i])[0]) if row[0]] 
 
 center=np.array(center)
 center.sort()
 #print(center)
 #Regarding the distance to the columns center, the boxes are arranged in respective order  

 finalboxes = []
 for i in range(np.shape(row)[0]):
     lis=[]
     for k in range(countcol):
        lis.append([])
     for j in range(np.shape(row[i])[0]):
           diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
           minimum = min(diff)
           indexing = list(diff).index(minimum)
           lis[indexing].append(row[i][j])
     finalboxes.append(lis) 

 print(np.shape(finalboxes)[0])
 #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
 outer=[] 

 outer.append(list(['number','name','fathername','house no','age','gender']))
 for i in range(np.shape(finalboxes)[0]):
     for j in range(np.shape(finalboxes[i])[0]):
         inner=[]
        
         if(np.shape(finalboxes[i][j])[0]==0):
             outer1=''
         else:
             for k in range(np.shape(finalboxes[i][j])[0]):
                 y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                 finalimg = bitnot[x:x+h, y:y+w]
                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                 border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                 resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                 dilation = cv2.dilate(resizing, kernel,iterations=1)
                 erosion = cv2.erode(dilation, kernel,iterations=2)
                
                 finalimg1 = bitnot[x:x+22, y+80:y+105]
                 resizing1 = cv2.resize(finalimg1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                 #plotting = plt.imshow(resizing1,cmap='gray')
                 #plt.show()
                 out3 = pytesseract.image_to_string(resizing1)
                 print(out3)
                 out = pytesseract.image_to_string(erosion,lang='hin')
                 out1 = pytesseract.image_to_string(erosion,lang="eng")
                 print(out)
                 print(out1)
                 a1=out1.split('\n')
                 inner.append(a1)
                
                 if(np.shape(out)==0):
                     out = pytesseract.image_to_string(erosion, config='--psm 3')
                 a7=out.split('\n')
                 a=[]
                 for x in a7:
                     
                    if x != "":
                     a.append(x)
                 toll=a[0]
                 print(toll[0:3])
                 if toll[0:3] == "नाम":
                    
                     a[4]=a[3]
                     a[3]=a[2]
                     a[2]=a[1]
                     a[1]=a[0]
                     print("amanl")
                     print(a[1])
                     print(a[2])
                     print(a[3])
                     print(a[4])
                     
                 t=0
                 try:
                  l1=a[1].replace("नाम","").replace(':','')
                 
                  k2=a[2].replace("पिता का नाम","").replace(':','')
             
                  l2=k2.replace("पति का नाम","").replace(':','')   
                  l3=a[3].replace("मकान संख्या","").replace(':','')
                  j45=a[4].replace(' लिंग :','लिंग').replace(':','')
                  j4=j45.split('लिंग')
                 
                  l4=j4[0].replace("आयु","").replace(':','').replace("।","1")
                
                 
                  l4=j4[0].replace("आयु","").replace(':','').replace("।","1")
                  l4=l4.replace("आयु","").replace(':','').replace("।","1")
                  #print(a)
                  #print(np.shape(j4))
                  l5=j4[1].replace("॥","1")
                  
            
                 
                
                  inner.append(l1)
                  inner.append(l2)
                  inner.append(l3)
                  inner.append(l4)
                  inner.append(l5)
                  print(inner)
                  outer.append(inner)
                  inner=[]
                 except:
                  print('corrupt data')   
                
 print(np.shape(outer))
 #Creating a dataframe of the generated OCR list
 arr=outer
 dataframe = pd.DataFrame(arr)
 print(dataframe)
 data = dataframe.style.set_properties(align="left")
 #Converting it in a excel-file
 data.to_excel("C:/Users/amank/Downloads/output/"+str(u)+".xlsx")
