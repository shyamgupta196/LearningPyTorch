'''

'''

import os
import numpy as np
from tqdm import tqdm
import cv2
# preparing our data 

REBUILD_DATA = True

class DogsVsCats:
    SIZE = 50
    CATS = 'PetImages/Cat'
    DOGS = 'PetImages/Dog'
    LABELS = {CATS:0,DOGS:1}
    training_data = []
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            # print(os.listdir(label))
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label,f)
                    # print(path)
                    img = cv2.imread(path,cv.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.SIZE,self.SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])
                    print(img)
                    break
                    if label ==  self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount+=1
                except Exception as e: 
                    pass
        np.random.shuffle(self.training_data)
        np.save('training_data.npy',self.training_data)
        print('cats:',self.catcount)
        print('dogs:',self.dogcount)

if REBUILD_DATA:
    DogsVsCats().make_training_data()


training_data = np.load('training_data.npy',allow_pickle=False)
print(len(training_data))

'''
well Idk why but the data is saved in  1kb file only 
i tried many things from sentdex suggestions but they didnt work
lets try some other method
'''
