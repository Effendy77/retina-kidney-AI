import pandas as pd
import os
import ntpath

# Fix filename extraction across Windows and Linux
def get_filename(path):
    # handles both / and \ based paths
    head, tail = ntpath.split(path)
    return tail if tail else ntpath.basename(head)


# ------------- CONFIGURATION -----------------
IMAGE_ROOT = "/home/fendy77/data/retina_images"
MASK_ROOT  = "/home/fendy77/projects/retina-kidney-AI/data/masks_raw_binary"

INPUT_CSV  = "/home/fendy77/projects/retina-kidney-AI/data/multimodal_master.csv"
OUTPUT_CSV = "/home/fendy77/projects/retina-kidney-AI/data/multimodal_master_with_paths.csv"
# ----------------------------------------------


df = pd.read_csv(INPUT_CSV)

# Find which column contains filenames
if "image_path" in df.columns:
    filename_col = "image_path"
elif "left_eye_filename" in df.columns:
    filename_col = "left_eye_filename"
else:
    raise ValueError("No image filename column found.")

# Extract clean filenames
df["clean_filename"] = df[filename_col].apply(get_filename)

# Build full correct paths
df["fundus_image"] = df["clean_filename"].apply(lambda x:
    os.path.join(IMAGE_ROOT, x)
)

df["vessel_mask"]  = df["clean_filename"].apply(lambda x:
    os.path.join(MASK_ROOT, x)
)

# Save
df.to_csv(OUTPUT_CSV, index=False)

print("Saved corrected CSV:", OUTPUT_CSV)
print(df[["clean_filename", "fundus_image", "vessel_mask"]].head())
