import json
import os
from glob import glob

# Define category names
categories = ["hime", "haru"]


def convert_to_yolo(annotation_file, output_dir):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    base_filename = os.path.splitext(os.path.basename(annotation_file))[0]
    yolo_annotation_file = os.path.join(output_dir, base_filename + '.txt')

    with open(yolo_annotation_file, 'w') as yolo_f:
        for shape in data['shapes']:
            points = shape['points']
            label = shape['label']

            # Convert label to category index
            class_idx = categories.index(label)

            # Get bounding box coordinates
            x_min = min(points[0][0], points[1][0])
            x_max = max(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            y_max = max(points[0][1], points[1][1])

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            yolo_f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")


def convert_all_json_to_yolo(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob(os.path.join(input_dir, '*.json'))
    for json_file in json_files:
        convert_to_yolo(json_file, output_dir)


# Specify input and output directories
input_directory = '/path/to/your/json/file/directory'
output_directory = '/set/your/txt/format/directory'

# Convert all JSON files
convert_all_json_to_yolo(input_directory, output_directory)
