# import kagglehub
import os 
import os
from PIL import Image


# Dataset
# path = kagglehub.dataset_download("anthonytherrien/dog-vs-cat") 

# train_path = "data/train"
# pictures = [file for file in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, file))]

# for index, picture in enumerate(pictures):
#   os.rename(os.path.join(train_path, picture), os.path.join(train_path, str(index) + ".jpg"))


# faces_path = "data/faces"
# face_files = sorted([file for file in os.listdir(faces_path) if os.path.isfile(os.path.join(faces_path, file))])

# for index, file_name in enumerate(face_files):
#   os.rename(
#     os.path.join(faces_path, file_name),
#     os.path.join(faces_path, f"{index}.png")
#   )

faces_path = "data/faces"
face_files = sorted([file for file in os.listdir(faces_path) if os.path.isfile(os.path.join(faces_path, file))])

for file_name in face_files:
  file_path = os.path.join(faces_path, file_name)
  base, ext = os.path.splitext(file_path)
  # Skip files that are already in JPG format
  if ext.lower() != ".jpg":
    with Image.open(file_path) as img:
      rgb_img = img.convert("RGB")
      new_file_path = base + ".jpg"
      rgb_img.save(new_file_path, "JPEG")
    os.remove(file_path)