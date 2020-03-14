
# coding: utf-8

# ## Convert .npy int .jpg
# * training data: folder structure as
#     - root_folder_name
#         - class 1
#             - file1
#             - file2
#         - class 2
#             - file1
#             - file2
# * test data: images under label_image folder

# In[ ]:

import numpy as np
import png
import os

file_path = "../Data/"
training_data = np.load(file_path+"tinyX.npy") #(26344, 3, 64, 64)
training_label = np.load(file_path+"tinyY.npy") #(26344,)
test_data = np.load(file_path+"tinyX_test.npy") #(6600, 3, 64, 64)

for i in range(training_data.shape[0]):
    directory = file_path + str(training_label[0])+"/";
    if not os.path.exists(directory):
        os.makedirs(directory)
    png.from_array(training_data[i].transpose(2,1,0),'RGB').save(file_path + str(training_label[0])+"/"+str(i)+".jpg")

directory = file_path + "label_image/";
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(test_data.shape[0]):
    png.from_array(training_data[i].transpose(2,1,0),'RGB').save(file_path + "label_image/"+str(i)+".jpg")
    


# In[ ]:




# In[ ]:



