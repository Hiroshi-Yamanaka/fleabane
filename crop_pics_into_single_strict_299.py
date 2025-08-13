import cv2
import os

def strict_crop_and_save_short_names(yolo_labels_dir, original_images_dir, output_dataset_dir, crop_size=(299, 299)):
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    for image_idx, image_filename in enumerate(image_files):
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_filename)
        image_path = os.path.join(original_images_dir, image_filename)
        
        if not os.path.exists(label_path):
            print(f"warn: there is no '{label_filename}',skipped")
            continue
            
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        
        with open(label_path, 'r') as f:
            for annotation_idx, line in enumerate(f.readlines()):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                center_x, center_y, _, _ = map(float, parts[1:])

                x_center = int(center_x * w)
                y_center = int(center_y * h)
                
                half_width = crop_size[0] // 2
                half_height = crop_size[1] // 2
                
                x1 = x_center - half_width
                y1 = y_center - half_height
                x2 = x1 + crop_size[0]
                y2 = y1 + crop_size[1]

                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                cropped_image = image[y1:y2, x1:x2]

                class_folder = os.path.join(output_dataset_dir, f'class_{class_id}')
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                image_id_padded = str(image_idx).zfill(3)
                annotation_id_padded = str(annotation_idx).zfill(2)
                class_id_padded = str(class_id).zfill(1)

                output_filename = f'{class_id_padded}_{image_id_padded}_{annotation_id_padded}.jpg'
                cv2.imwrite(os.path.join(class_folder, output_filename), cropped_image)
                

# example
yolo_labels_dir = '/path/to/your/label/directory'
original_images_dir = '/path/to/your/picture/directory'
output_dataset_dir = '/set/your/own/output/directory'

strict_crop_and_save_short_names(yolo_labels_dir, original_images_dir, output_dataset_dir)
