import os
import pandas as pd
from PIL import Image

from datasets import load_dataset, concatenate_datasets, Dataset, Image as ImageHF
from huggingface_hub import hf_hub_download


Image.MAX_IMAGE_PIXELS = None # scans exceed default MAX_IMAGE_PIXELS and trigger a DecompressionBombError

def verify_format(filename):
    valid = True
    msg = ''
    with open(f'./raw data/all_files/{filename}', 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    if len(lines) != 13: 
        msg += f"Error in {filename}: expected length to be 13, received {len(lines)} instead.\n"
        valid = False
    if lines[0] != f'LM={len(lines) - 2}':
        msg += f"Error in {filename}: expected first line to be LM={len(lines) - 2}, instead, first line is {lines[0]}\n"
        valid = False
    if not lines[-1].startswith('IMAGE='):
        if lines[-1].startswith('ID='): 
            msg += f"Warning: final line in {filename} is {lines[-1]}\n"
            if not lines[-2].startswith('IMAGE='):
                msg += "Error: IMAGE= not found in end of file\n"
                valid = False
        else: 
            msg += f"Error: IMAGE= not found in end of file\n"
            valid = False
    
    return valid, msg

valid_TPS_files = []

for filename in sorted(os.listdir('./raw data/all_files')):
    valid, msg = verify_format(filename)
    if valid: 
        valid_TPS_files.append(filename)
    else: 
        print(msg, end='')

def parse_TPS(data):
    coords = []
    endline = -1
    if data[endline].startswith('ID'): endline = -2
    for coord_str in data[1:endline]:
        coord_x, coord_y = coord_str.split(' ')
        coords.append({'x': float(coord_x), 'y': float(coord_y)})
    distance = ((coords[1]['x'] - coords[0]['x']) ** 2 + (coords[1]['y'] - coords[0]['y']) ** 2)**0.5
    image_file = lines[endline].replace('IMAGE=','')
    return {'distance': distance, 'coords': coords[2:], 'image_file': image_file}

TPS_data = []
for filename in valid_TPS_files:
    with open(f'./raw data/all_files/{filename}', 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    data = parse_TPS(lines)
    if 'toe' in filename:
        data['type'] = 'toe'
    if 'finger' in filename:
        data['type'] = 'finger'
    TPS_data.append(data)

def get_bounding_box(data):
    coords = data['coords']
    min_x = 10**10
    min_y = 10**10
    max_x = -1.0
    max_y = -1.0
    for coord in coords:
        x = coord['x']
        y = coord['y']
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y
    return min_x, max_x, min_y, max_y

padding = 50
data = TPS_data[1]
bbox = get_bounding_box(data)
bbox_width = bbox[1] - bbox[0]
bbox_height = bbox[3] - bbox[2]
bbox_dim = max([400, bbox_width, bbox_height])

if data['image_file'] in os.listdir('./scans/spring_2024/'): 
    filename = f"./scans/spring_2024/{data['image_file']}"
elif data['image_file'] in os.listdir('./scans/fall_2024/'): 
    filename = f"./scans/fall_2024/{data['image_file']}"
full_img = Image.open(filename)
width, height = full_img.size
bbox_center = [0.5 * (bbox[0] + bbox[1]), height - 0.5 * (bbox[2] + bbox[3])]
crop_bbox = (
    bbox_center[0] - bbox_dim / 2 - padding,
    bbox_center[1] - bbox_dim / 2 - padding,
    bbox_center[0] + bbox_dim / 2 + padding,
    bbox_center[1] + bbox_dim / 2 + padding,
)

crops = {
    'finger': {},
    'toe': {}
}
image_sizes = {}
padding = 50
repo_id = "ttaylor99/HAAG_Lizard_Toepad_Scans" 

for data in TPS_data:
    print(f"Processing {data['image_file']}")
    bbox = get_bounding_box(data)
    bbox_width = bbox[1] - bbox[0]
    bbox_height = bbox[3] - bbox[2]
    
    # if data['image_file'] in os.listdir('./scans/spring_2024/'): 
    #     local_filepath = f"./scans/spring_2024/{data['image_file']}"
    # elif data['image_file'] in os.listdir('./scans/fall_2024/'): 
    #     local_filepath = f"./scans/fall_2024/{data['image_file']}"
    # else:
    #     print(f"File not found: {data['image_file']}")
    #     break

    local_filepath = hf_hub_download(repo_id=repo_id, filename=data['image_file'], repo_type="dataset")
    
    full_img = Image.open(local_filepath)
    width, height = full_img.size
    image_sizes[data['image_file']] = (width, height)
    bbox_center = [0.5 * (bbox[0] + bbox[1]), height - 0.5 * (bbox[2] + bbox[3])]
    crop_bbox = (
        bbox_center[0] - bbox_width / 2 - padding,
        bbox_center[1] - bbox_height / 2 - padding,
        bbox_center[0] + bbox_width / 2 + padding,
        bbox_center[1] + bbox_height / 2 + padding,
    )
    crops[data['type']][data['image_file']] = crop_bbox
    cropped_img = full_img.crop(crop_bbox)
    cropped_img_filename = f"./cropped/{data['type']}/{data['image_file']}"
    cropped_img.save(cropped_img_filename)
    print(f"file saved at {cropped_img_filename}")
    os.remove(local_filepath)

out_lines = ''
for i, data in enumerate([_ for _ in TPS_data if _['type'] == 'finger']):
    crop = crops['finger'][data['image_file']]
    width, height = image_sizes[data['image_file']]
    out_lines += 'LM=9\n'
    for coord in data['coords']:
        out_lines += f"{coord['x'] - crop[0]} {coord['y'] + crop[3] - height}\n"
    # out_lines += f"IMAGE={data['image_file'].replace('.jpg','-' + data['type'] + '.jpg')}\n"
    out_lines += f"IMAGE={data['image_file']}\n"
    out_lines += f"ID={i}\n"

with open('all.tps', 'w') as file:
    file.write(out_lines)