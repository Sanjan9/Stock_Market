from PIL import Image
import os

# Folder where PNG images are stored
image_folder = "."  # Make sure you're in the correct directory

# Get only the latest images
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith("_prediction.png")])

# Ensure there are images to process
if not image_files:
    print("❌ No new prediction images found. Make sure your script generated new images.")
    exit()

# Open images and convert them to PDF format
image_list = [Image.open(os.path.join(image_folder, img)).convert("RGB") for img in image_files]

# Save all images into a single PDF
pdf_filename = "stock_predictions.pdf"
image_list[0].save(pdf_filename, save_all=True, append_images=image_list[1:])

print(f"\n✅ PDF saved as '{pdf_filename}' with the latest stock predictions.")

