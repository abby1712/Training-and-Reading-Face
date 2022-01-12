import face_recognition
import cv2 
import os

Encodings=[]
#learnings of the Known Faces
Names=[]
#Names of The Known Faces
image_dir='/home/abby/Desktop/PyPro/faceRecognizer/demoImages/known'
# the Folder which has all the Images
# to Walk through all the files of the Folder(Known0)
for root,dirs,files in os.walk(image_dir):
    print(files)
    for file in files:
        path=os.path.join(root,file) # Joining the root found in the Image directory with The file name\
        print(path)
        # Getting The name of the person from then file
        name =os.path.splitext(file)[0]
        print(name)
        person=face_recognition.load_image_file(path) # loading the person's Pic to the variable "person"
        encoding=face_recognition.face_encodings(person)[0] # Learning the "Person"'s Face into Variable "encoding"
        Encodings.append(encoding) #appending the learning of the person to the entire list
        Names.append(name)  #appending the Name of the person to the entire list

print(Names)

with open('train.pkl','wb')as f:
        pickle.dump(Names,f)
        pickle.dump(Encodings,f)

  
