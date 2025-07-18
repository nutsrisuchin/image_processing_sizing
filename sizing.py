import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import tempfile
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from fpdf import FPDF
import io

# --- PDF Report Generation ---

def generate_report(analysis_data, annotated_image, raw_img, final_img, diam_hist_fig, circ_hist_fig, filename):
    """Generates a PDF report using fpdf2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Analysis Report for: {filename}", 0, 1, "C")
    pdf.ln(10)

    # --- Summary Stats ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Summary Statistics", 0, 1)
    
    all_diameters = [d['h_diam'] if d['h_diam'] >= d['v_diam'] else d['v_diam'] for d in analysis_data]
    all_circularities = [d['circularity'] for d in analysis_data if d['circularity'] > 0]
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, f"- Total Foams Detected: {len(analysis_data)}", 0, 1)
    if all_diameters:
        pdf.cell(0, 5, f"- Mean Diameter: {np.mean(all_diameters):.2f} µm", 0, 1)
        pdf.cell(0, 5, f"- Median Diameter: {np.median(all_diameters):.2f} µm", 0, 1)
        pdf.cell(0, 5, f"- Std. Deviation (Diameter): {np.std(all_diameters):.2f} µm", 0, 1)
    if all_circularities:
        pdf.cell(0, 5, f"- Mean Circularity: {np.mean(all_circularities):.2f}", 0, 1)
    pdf.ln(10)

    # --- Images ---
    def add_image_to_pdf(pdf_obj, header, image, fig=None):
        pdf_obj.set_font("Helvetica", "B", 12)
        pdf_obj.cell(0, 10, header, 0, 1)
        
        # Save image/figure to a byte stream
        img_byte_arr = io.BytesIO()
        if fig:
            fig.savefig(img_byte_arr, format='PNG', bbox_inches='tight')
        else:
            image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Add to PDF
        pdf_obj.image(img_byte_arr, w=180)
        pdf_obj.ln(5)

    # ADDED: Include detection and raw measurement images in the report
    add_image_to_pdf(pdf, "Object Detection Results", annotated_image)
    add_image_to_pdf(pdf, "Raw Diameter Measurements", raw_img)
    add_image_to_pdf(pdf, "Final Diameter Measurements", final_img)
    add_image_to_pdf(pdf, "Diameter Distribution", None, fig=diam_hist_fig)
    add_image_to_pdf(pdf, "Circularity Distribution", None, fig=circ_hist_fig)

    # --- Raw Data Table ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Detailed Measurement Data", 0, 1)
    pdf.set_font("Helvetica", "B", 10)
    
    # Table Header
    col_width = pdf.w / 5.5
    pdf.cell(col_width, 10, "Object #", 1, 0, "C")
    pdf.cell(col_width, 10, "H-Diam (µm)", 1, 0, "C")
    pdf.cell(col_width, 10, "V-Diam (µm)", 1, 0, "C")
    pdf.cell(col_width, 10, "Final Diam (µm)", 1, 0, "C")
    pdf.cell(col_width, 10, "Circularity", 1, 1, "C")
    
    # Table Rows
    pdf.set_font("Helvetica", "", 10)
    for item in analysis_data:
        final_diam = max(item['h_diam'], item['v_diam'])
        pdf.cell(col_width, 10, str(item['number']), 1, 0, "C")
        pdf.cell(col_width, 10, f"{item['h_diam']:.2f}", 1, 0, "C")
        pdf.cell(col_width, 10, f"{item['v_diam']:.2f}", 1, 0, "C")
        pdf.cell(col_width, 10, f"{final_diam:.2f}", 1, 0, "C")
        pdf.cell(col_width, 10, f"{item['circularity']:.2f}", 1, 1, "C")

    # FIX: Use pdf.output() which returns bytes for fpdf2
    return pdf.output()

# --- Core Functions from your script ---

def detect_with_ultralytics(model_path, image_path, conf_threshold=0.8, iou_threshold=0.3):
    """
    Part 1: Uses the native ultralytics library for detection.
    This function loads the .pt model, runs prediction, and extracts the
    bounding box data.
    """
    st.write("--- Starting Part 1: Object Detection ---")
    
    # Load the YOLOv8 model
    st.write(f"Loading model from: `{os.path.basename(model_path)}`")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure 'best.pt' is in the same directory as the script.")
        return None, None, None
        
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None, None, None

    # Run Prediction
    st.write(f"Running prediction on: `{os.path.basename(image_path)}`")
    
    # Create a temporary directory to save results
    temp_dir = tempfile.mkdtemp()
    try:
        results = model.predict(source=image_path, conf=conf_threshold, iou=iou_threshold, save=True, project=temp_dir)
        
        result = results[0] # Get the result for the first image

        saved_image_directory = result.save_dir
        st.success(f"Ultralytics saved the annotated image successfully.")

        saved_image_files = [f for f in os.listdir(saved_image_directory) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))]
        if not saved_image_files:
            st.error("Could not find the annotated image saved by Ultralytics.")
            return None, None, None

        annotated_image_path = os.path.join(saved_image_directory, saved_image_files[0])
        
        with Image.open(annotated_image_path) as img:
            annotated_image = img.copy()

        final_boxes = result.boxes.xyxy.cpu().numpy()
        final_scores = result.boxes.conf.cpu().numpy()

        return final_boxes, final_scores, annotated_image
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def visualize_and_measure(boxes, original_image, scale_mm):
    """
    Part 2: Creates visualizations and returns detailed analysis data.
    """
    st.write("\n--- Starting Part 2: Measurement and Visualization ---")

    if scale_mm == '2mm':
        PIXELS_FOR_SCALE = 730.0
        MICRONS_IN_SCALE = 2000.0
    elif scale_mm == '1mm':
        PIXELS_FOR_SCALE = 365.0
        MICRONS_IN_SCALE = 1000.0
    else:
        st.error("Invalid scale selected.")
        return None, None, None, None

    MICRONS_PER_PIXEL = MICRONS_IN_SCALE / PIXELS_FOR_SCALE
    st.info(f"Calibration: Using {MICRONS_PER_PIXEL:.2f} µm/pixel for a {scale_mm} scale.")

    if len(boxes) == 0:
        st.warning("No objects to visualize or measure.")
        return None, None, None, None

    st.write(f"Found {len(boxes)} objects to process.")

    # Attempt to load a font with good unicode support
    font_size = 25
    try:
        # On Streamlit Cloud, you might need to provide a font file in your repo
        # or rely on a common system font.
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        st.warning("DejaVuSans.ttf font not found. Trying Arial...")
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            st.warning("Arial.ttf font not found. Using basic PIL font. Special characters may not render.")
            font = ImageFont.load_default()


    analysis_data = []
    for i, box in enumerate(boxes):
        object_number = i + 1
        x1, y1, x2, y2 = box
        h_diameter_px = x2 - x1
        v_diameter_px = y2 - y1
        h_diameter_um = h_diameter_px * MICRONS_PER_PIXEL
        v_diameter_um = v_diameter_px * MICRONS_PER_PIXEL
        center_x = x1 + h_diameter_px / 2
        center_y = y1 + v_diameter_px / 2
        circularity = 0
        # --- REVISED FORMULA ---
        # Calculate circularity as smaller diameter / larger diameter.
        # A value of 1 is a perfect circle.
        if max(h_diameter_um, v_diameter_um) > 0:
            circularity = min(h_diameter_um, v_diameter_um) / max(h_diameter_um, v_diameter_um)
        analysis_data.append({
            'number': object_number, 'box': (x1, y1, x2, y2),
            'h_diam': h_diameter_um, 'v_diam': v_diameter_um,
            'circularity': circularity, 'center_x': center_x, 'center_y': center_y
        })

    # --- Create visualization images ---
    image_with_raw_measurements = original_image.copy()
    draw_raw = ImageDraw.Draw(image_with_raw_measurements)
    image_with_final_measurements = original_image.copy()
    draw_final = ImageDraw.Draw(image_with_final_measurements)
    image_with_circularity = original_image.copy()
    draw_circularity = ImageDraw.Draw(image_with_circularity, "RGBA")

    for data in analysis_data:
        x1, y1, x2, y2 = data['box']
        center_x, center_y = data['center_x'], data['center_y']
        
        # Raw measurements
        draw_raw.line([(x1, center_y), (x2, center_y)], fill="cyan", width=3)
        h_text = f"{data['h_diam']:.1f}"
        draw_raw.text((center_x, center_y - 10), h_text, fill="cyan", font=font, anchor="mb")
        draw_raw.line([(center_x, y1), (center_x, y2)], fill="magenta", width=3)
        v_text = f"{data['v_diam']:.1f}"
        draw_raw.text((center_x + 10, center_y), v_text, fill="magenta", font=font, anchor="lm")

        # Final measurements
        object_number = data['number']
        line_color, line_width, tick_size = "red", 5, 12
        if data['h_diam'] >= data['v_diam']:
            text = f"{object_number}: {data['h_diam']:.1f} µm"
            draw_final.line([(x1, center_y), (x2, center_y)], fill=line_color, width=line_width)
            draw_final.line([(x1, center_y - tick_size), (x1, center_y + tick_size)], fill=line_color, width=line_width)
            draw_final.line([(x2, center_y - tick_size), (x2, center_y + tick_size)], fill=line_color, width=line_width)
            draw_final.text((center_x, center_y - tick_size), text, fill=line_color, font=font, anchor="mb")
        else:
            text = f"{object_number}: {data['v_diam']:.1f} µm"
            draw_final.line([(center_x, y1), (center_x, y2)], fill=line_color, width=line_width)
            draw_final.line([(center_x - tick_size, y1), (center_x + tick_size, y1)], fill=line_color, width=line_width)
            draw_final.line([(center_x - tick_size, y2), (center_x + tick_size, y2)], fill=line_color, width=line_width)
            draw_final.text((center_x + tick_size, center_y), text, fill=line_color, font=font, anchor="lm")

        # Circularity
        text = f"{data['number']}: {data['circularity']:.2f}"
        text_color = "#FDB813" # Using the user-provided gold color
        try:
            text_bbox = draw_circularity.textbbox((center_x, center_y), text, font=font, anchor="mm")
            padding = 5
            bg_bbox = (text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding)
            draw_circularity.rectangle(bg_bbox, fill=(0, 0, 0, 180))
            draw_circularity.text((center_x, center_y), text, fill=text_color, font=font, anchor="mm")
        except TypeError:
            draw_circularity.text((center_x - 15, center_y - 10), text, fill=text_color, font=font)

    return image_with_raw_measurements, image_with_final_measurements, image_with_circularity, analysis_data

def create_diameter_histogram(diameters):
    """Creates, styles, and returns a histogram figure for diameter data."""
    st.write("\n--- Creating Diameter Distribution Histogram ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    n, bins, patches = ax1.hist(diameters, bins=20, color='red', alpha=0.7, rwidth=0.85)
    ax1.set_xlabel('Diameter (µm)', fontsize=14)
    ax1.set_ylabel('Number of Foams (Frequency)', fontsize=14, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    
    ax2 = ax1.twinx()
    cumulative_percentage = 100 * np.cumsum(n) / n.sum()
    ax2.plot(bins[1:], cumulative_percentage, color='green', linestyle='-', marker='.', ms=8)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.set_ylabel('Cumulative Percentage', fontsize=14, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 100)

    plt.title('Distribution of Foam Diameters', fontsize=18, fontweight='bold')
    
    mean_val, median_val, std_dev = np.mean(diameters), np.median(diameters), np.std(diameters)
    stats_text = (
        f'Total Foams: {len(diameters)}\n'
        f'Mean Diameter: {mean_val:.2f} µm\n'
        f'Median Diameter: {median_val:.2f} µm\n'
        f'Std. Deviation: {std_dev:.2f} µm'
    )
    
    plt.subplots_adjust(bottom=0.25)
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.05, stats_text, ha='center', fontsize=12, bbox=props)

    return fig

def create_circularity_histogram(circularities):
    """Creates, styles, and returns a histogram figure for circularity data."""
    st.write("\n--- Creating Circularity Distribution Histogram ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    valid_circularities = [c for c in circularities if c > 0]
    
    ax.hist(valid_circularities, bins=20, color='blue', alpha=0.7, rwidth=0.85)
    # --- REVISED LABEL ---
    ax.set_xlabel('Circularity (Smaller/Larger Diameter)', fontsize=14)
    ax.set_ylabel('Number of Foams (Frequency)', fontsize=14)
    
    plt.title('Distribution of Foam Circularity', fontsize=18, fontweight='bold')
    
    mean_val, median_val, std_dev = np.mean(valid_circularities), np.median(valid_circularities), np.std(valid_circularities)
    stats_text = (
        f'Total Foams: {len(valid_circularities)}\n'
        f'Mean Circularity: {mean_val:.2f}\n'
        f'Median Circularity: {median_val:.2f}\n'
        f'Std. Deviation: {std_dev:.2f}'
    )
    
    plt.subplots_adjust(bottom=0.25)
    props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5)
    fig.text(0.5, 0.05, stats_text, ha='center', fontsize=12, bbox=props)
    
    return fig

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🔬 Foam Cell Diameter Analysis Tool")
st.info("Upload your SEM image. The app will use the `best.pt` model from the repository. Adjust the parameters and click 'Analyze Image'.")

with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    uploaded_image = st.file_uploader("1. Upload SEM Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    st.markdown("---")
    st.write("2. Model: `best.pt` (loaded automatically)")
    st.markdown("---")
    selected_scale = st.selectbox("3. Select Image Scale", options=['2mm', '1mm'], index=0)
    conf_slider = st.slider("4. Confidence Threshold", 0.0, 1.0, 0.75, 0.05)
    iou_slider = st.slider("5. IOU Threshold", 0.0, 1.0, 0.5, 0.05)
    analyze_button = st.button("Analyze Image", use_container_width=True)

if analyze_button and uploaded_image:
    with st.spinner('Processing... Please wait.'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as tmp_image:
            tmp_image.write(uploaded_image.getvalue())
            image_path = tmp_image.name
        
        # --- MODEL PATH IS NOW HARDCODED ---
        model_path = "best.pt"

        original_pil_image = Image.open(image_path).convert("RGB")
        boxes, scores, annotated_image = detect_with_ultralytics(model_path, image_path, conf_slider, iou_slider)

        if boxes is not None and annotated_image is not None and len(boxes) > 0:
            st.success(f"Detection complete! Found {len(boxes)} objects.")
            
            raw_img, final_img, circ_img, analysis_data = visualize_and_measure(boxes, original_pil_image, selected_scale)

            st.header("📊 Analysis Results")

            st.subheader("1. Object Detection")
            st.image(annotated_image, caption="YOLOv8 Detection Results", use_column_width=True)
            
            st.subheader("2. Final Diameter (µm)")
            st.image(final_img, caption="Final Diameter Measurements (Largest of H or V)", use_column_width=True)

            st.subheader("3. Raw Diameters (H & V)")
            st.image(raw_img, caption="Raw Horizontal (cyan) and Vertical (magenta) Diameters", use_column_width=True)

            st.subheader("4. Circularity")
            # --- REVISED CAPTION ---
            st.image(circ_img, caption="Circularity (Smaller Diameter / Larger Diameter)", use_column_width=True)

            with st.expander("Show Detailed Measurement Data"):
                for item in analysis_data:
                    st.text(
                        f"- Object {item['number']}: H-Diam={item['h_diam']:.1f}µm, "
                        f"V-Diam={item['v_diam']:.1f}µm, Circularity={item['circularity']:.2f}"
                    )
            
            st.header("📈 Statistical Analysis")
            all_diameters = [d['h_diam'] if d['h_diam'] >= d['v_diam'] else d['v_diam'] for d in analysis_data]
            all_circularities = [d['circularity'] for d in analysis_data]

            fig_diam = create_diameter_histogram(all_diameters)
            st.pyplot(fig_diam)

            fig_circ = create_circularity_histogram(all_circularities)
            st.pyplot(fig_circ)

            # --- PDF Generation and Download Button ---
            st.header("📄 Download Report")
            # ADDED: Pass the new images to the report generator
            pdf_bytes = generate_report(
                analysis_data,
                annotated_image,
                raw_img,
                final_img,
                fig_diam,
                fig_circ,
                uploaded_image.name
            )
            
            # FIX: Convert bytearray from fpdf2 to bytes for Streamlit
            st.download_button(
                label="Download Full Report (PDF)",
                data=bytes(pdf_bytes),
                file_name=f"{os.path.splitext(uploaded_image.name)[0]}_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        else:
            st.error("Analysis failed. No objects were detected or an error occurred in Part 1.")

        os.remove(image_path)

elif analyze_button:
    st.warning("Please upload an image to start the analysis.")
