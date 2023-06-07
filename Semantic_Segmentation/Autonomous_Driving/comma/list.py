import os 

path = 'All_Data/June-5/retrain'

image_list = [line + '\n' for line in os.listdir(os.path.join(path, 'imgs'))]
with open(os.path.join(path, 'files'), 'w') as file:
    file.writelines(image_list) 

