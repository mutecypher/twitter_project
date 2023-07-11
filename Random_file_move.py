import os
import random
import shutil




def move_files(files, source, destination):
  for file in files:
    source_path = os.path.join(source, file)
    destination_path = os.path.join(destination, file)
    shutil.move(source_path, destination_path)
 
source_dir = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train/cat_in_mouth'
destination_dir = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/test/cat_in_mouth'
 
file_list = os.listdir(source_dir)

num_files_to_move = int(0.2 * len(file_list))

files_to_move = random.sample(file_list, num_files_to_move) 
move_files(files_to_move, source_dir, destination_dir)

source_dir = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train/cat'
destination_dir = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/test/cat'
 
file_list = os.listdir(source_dir)

num_files_to_move = int(0.2 * len(file_list))

files_to_move = random.sample(file_list, num_files_to_move) 
move_files(files_to_move, source_dir, destination_dir)

source_dir = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train/non_cats'
destination_dir = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/test/non_cats'
 
file_list = os.listdir(source_dir)

num_files_to_move = int(0.2 * len(file_list))

files_to_move = random.sample(file_list, num_files_to_move) 
move_files(files_to_move, source_dir, destination_dir)

