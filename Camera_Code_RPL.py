import cv2
import numpy as np

def measure_water_levels(frame, num_buckets=5, pixels_per_mm=2.0):
    """
    Analyzes a live video frame to detect water levels in side-by-side transparent buckets.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 3. Divide the screen into equal vertical columns
    roi_width = width // num_buckets
    fluid_heights_mm = []

    for i in range(num_buckets):
        # Define the left and right X-coordinates for this specific bucket
        x_start = i * roi_width
        x_end = (i + 1) * roi_width
        
        # SAFETY MARGIN: Ignore the physical plastic walls of the bucket
        # We cut off the outer 15% of the column so the algorithm only looks at the center water
        margin = int(roi_width * 0.5)
        roi_edges = edges[:, x_start + margin : x_end - margin]
        
        # Find the coordinates of every white pixel (detected edge) in this column
        y_coords, x_coords = np.where(roi_edges == 255)
        
        if len(y_coords) > 0:
            # NOISE REJECTION: Take the 5th percentile of the top pixels to find the solid meniscus line.
            # (Avoids reading a splash or speck of dust as the water level)
            y_meniscus = int(np.percentile(y_coords, 5))
            
            # Calculate pixel height from the BOTTOM of the frame
            pixel_height = height - y_meniscus
            
            # Convert to physical engineering units
            physical_height_mm = pixel_height / pixels_per_mm
            fluid_heights_mm.append(physical_height_mm)
            
            # --- DRAW THE VISUAL OVERLAY ---
            # Draw a horizontal green line exactly at the detected water level
            cv2.line(frame, (x_start, y_meniscus), (x_end, y_meniscus), (0, 255, 0), 2)
            
            # Puts height above line
            text = f"{physical_height_mm:.1f} mm"
            cv2.putText(frame, text, (x_start + margin, y_meniscus - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # If no edges are detected, the bucket is empty
            fluid_heights_mm.append(0.0)
            cv2.putText(frame, "Empty", (x_start + margin, height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Draw vertical blue boundaries between the buckets for visual debugging
        cv2.line(frame, (x_end, 0), (x_end, height), (255, 0, 0), 1)

    return frame, edges, fluid_heights_mm


# --- WINDOWS EXECUTION ENGINE ---
if __name__ == "__main__":
    print("Starting Camera")
    
    # Using '0' to select the first USB/Internal Webcam. 
    # cv2.CAP_DSHOW forces Windows to open the camera instantly without the typical OS lag.
    cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2) 

    if not cap.isOpened():
        print("Windows could not find webcam at index 0.")
        exit()

    print("Camera Active. Press 'o' to stop.")

    while True:
        # Read the live frame from the hardware
        ret, current_frame = cap.read()
        
        if not ret:
            print("Camera feed errored")
            break
        frame = current_frame[215:290, 475:700]
        # Run the tracking function 
        # (Tune your number of buckets and your pixels-to-mm ratio here)
        processed_frame, edge_map, measurements = measure_water_levels(
            frame, 
            num_buckets=3,       # Change to 3 or 5 depending on your setup
            pixels_per_mm=5    # Update this ratio after your ruler calibration
        )
        
        # Display the live video feed and the internal Canny engine side-by-side
        cv2.imshow("Live Tracking Feed", processed_frame)
        cv2.imshow("Canny Edge Map", edge_map)

        # off protocol
        if cv2.waitKey(1) & 0xFF == ord('o'):
            print("camera off")
            break

    # Safely release the USB port and close Windows
    cap.release()
    cv2.destroyAllWindows()