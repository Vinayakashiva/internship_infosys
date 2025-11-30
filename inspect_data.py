import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ====================================================================
# --- ⚠️ Configuration: UPDATE THESE PATHS ⚠️ ---
# Set the path to the perfect image and the defective image pair.
# These paths are based on the structure you provided:
# Template: ./PCB_USED/01.JPG
# Test: ./Missing_hole/01_missing_hole_01.JPG
# ====================================================================

TEMPLATE_PATH = r"C:\Users\vinay\PycharmProjects\internship_infosys\PCB_DATASET\PCB_USED\01.JPG"
TEST_PATH = r"C:\Users\vinay\PycharmProjects\internship_infosys\PCB_DATASET\images\Missing_hole\01_missing_hole_01.jpg"
OUTPUT_DIR = "./module1_results"  # Directory to save the outputs


def process_pcb_for_defects(temp_path, test_path, save_dir):
    """
    Executes Module 1 tasks: Image preprocessing, subtraction, Otsu's thresholding,
    and morphological filtering to generate defect difference maps and masks.
    """

    # 1. Load Images and Preprocessing (Task: Cleaned and aligned dataset)
    # Load in color, as we will draw contours on the color image later.
    template = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    test = cv2.imread(test_path, cv2.IMREAD_COLOR)

    if template is None or test is None:
        print("\n❌ ERROR: Could not load template or test image.")
        print("Please check if the file paths are correct:")
        print(f"Template Path: {temp_path}")
        print(f"Test Path: {test_path}")
        return

    # Convert to Grayscale for reliable subtraction
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    print("✅ Images loaded and converted to grayscale.")

    # 2. Image Subtraction (Task: Defect difference maps)
    # Calculates the absolute difference to find areas where pixel intensity differs.
    diff_map = cv2.absdiff(gray_template, gray_test)

    # 3. Thresholding (Task: Highlight defect regions using Otsu’s method)
    # Otsu's method finds the best threshold automatically.
    # The output is the binary defect mask.
    ret, defect_mask_otsu = cv2.threshold(
        diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"✅ Otsu's method applied. Optimal Threshold (T) found: {ret}")

    # 4. Filters (Task: Use filters to highlight defect regions)
    # Define a kernel for morphological operations (5x5 is a good standard size)
    kernel = np.ones((5, 5), np.uint8)

    # Opening (Erosion then Dilation): Removes small isolated noise (speckles).
    mask_opened = cv2.morphologyEx(defect_mask_otsu, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing (Dilation then Erosion): Fills small gaps/holes within the defect regions.
    final_defect_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    print("✅ Morphological filtering applied (Opening and Closing).")

    # --- Generate Deliverables (Task: Sample defect-highlighted images) ---
    os.makedirs(save_dir, exist_ok=True)

    # Find contours (boundaries) of the detected defect regions
    # RETR_EXTERNAL retrieves only the outermost contours
    contours, _ = cv2.findContours(final_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original test image to draw the findings on
    defect_highlighted = test.copy()

    # Draw green contours around the detected defects (Evaluation: Accurate defect mask generation)
    cv2.drawContours(defect_highlighted, contours, -1, (0, 255, 0), 3)  # Green color (0, 255, 0), thickness 3

    # Save all deliverables
    base_name = os.path.splitext(os.path.basename(test_path))[0]
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_diffmap.png"), diff_map)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_mask_final.png"), final_defect_mask)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_highlighted_final.png"), defect_highlighted)

    # Display results for immediate inspection
    titles = ['Grayscale Difference Map', 'Final Binary Defect Mask', 'Defect Highlighted Test Image']
    images = [diff_map, final_defect_mask, cv2.cvtColor(defect_highlighted, cv2.COLOR_BGR2RGB)]

    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        # Use 'gray' colormap for the first two binary/grayscale images
        cmap = 'gray' if i < 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"\n✅ Module 1 successfully completed. Deliverables saved in: {save_dir}")


# Execute the function
process_pcb_for_defects(TEMPLATE_PATH, TEST_PATH, OUTPUT_DIR)