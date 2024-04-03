import cv2
import numpy as np

def generate_modulation_patterns(frame_count, height, width):
    """Generate simple binary modulation patterns."""
    patterns = np.zeros((frame_count, height, width), dtype=np.float32)
    for i in range(frame_count):
        patterns[i, :, :] = np.random.rand(height, width) > 0.5
    return patterns

def video_to_sci_encoder(video_path, output_path, frame_count=10, start_time_msec=0):
    """Convert a video to an SCI encoded snapshot, starting from start_time_msec."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Set the start position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Generate modulation patterns
    patterns = generate_modulation_patterns(frame_count, height, width)

    # Prepare an empty frame for the SCI snapshot
    sci_snapshot = np.zeros((height, width), dtype=np.float32)

    # Process video frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video.")
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        
        # Modulate the frame
        modulated_frame = gray_frame * patterns[i]
        
        # Add the modulated frame to the SCI snapshot
        sci_snapshot += modulated_frame

    # Normalize the SCI snapshot
    sci_snapshot = sci_snapshot / frame_count * 255
    sci_snapshot = sci_snapshot.astype(np.uint8)

    # Save the SCI snapshot
    cv2.imwrite(output_path, sci_snapshot)
    
    # Release the video capture object
    cap.release()
    print("SCI snapshot saved to", output_path)

# Example usage
video_path = "input/video.mp4"
output_path = "output/output.png"
start_time_msec = 10000  # Start at 10 seconds into the video
frame_count = 10
video_to_sci_encoder(video_path, output_path, frame_count, start_time_msec=start_time_msec)
