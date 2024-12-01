import os
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('models/best.pt')

# Specify the path to your test images
dataset_folder = 'bone-cancer-detection--1/test/normal'

# Make predictions
results = model.predict(source=dataset_folder, conf=0.25)

# Create output directory for annotated images
output_folder = 'classification_results'
os.makedirs(output_folder, exist_ok=True)

# Process and display results
for result in results:
    img_path = result.path  # Get the image path
    img = cv2.imread(img_path)  # Load the image

    # Get the classification label and confidence
    class_id = result.probs.top1  # Class ID with the highest probability
    confidence = result.probs.top1conf  # Confidence of the class
    label = f"{model.names[class_id]}: {confidence:.2f}"  # Class name and confidence

    # Annotate the image with the classification result
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the annotated image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, img)

    # Display the annotated image (optional)
    cv2.imshow('Annotated Image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
