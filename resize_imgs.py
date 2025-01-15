import os
from PIL import Image, ImageOps
import shutil

def resize_image(image_path, output_path, target_size=2000):
    # Open the image
    with Image.open(image_path) as img:
        # Apply orientation based on EXIF data
        img = ImageOps.exif_transpose(img)
        
        # Get original dimensions
        width, height = img.size
        
        # Calculate new dimensions while preserving aspect ratio
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save the resized image, preserving EXIF data
        try:
            exif = img.info.get('exif', None)
            if exif:
                resized_img.save(output_path, quality=95, optimize=True, exif=exif)
            else:
                resized_img.save(output_path, quality=95, optimize=True)
        except Exception:
            # If saving with EXIF fails, save without it
            resized_img.save(output_path, quality=95, optimize=True)

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clear output folder if it exists
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    
    # Copy the entire folder structure
    shutil.copytree(input_folder, output_folder, dirs_exist_ok=True)
    
    # Process all images in the folder
    for root, dirs, files in os.walk(output_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, filename)
                try:
                    resize_image(file_path, file_path)
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = "fotografia"
    output_folder = "fotografia_resized"
    
    process_folder(input_folder, output_folder)
    print("Resizing complete!")
