import cv2
import numpy as np
import os

def generate_modulation_patterns(frame_count, height, width):
    """Generate simple binary modulation patterns."""
    patterns = np.zeros((frame_count, height, width), dtype=np.float32)
    for i in range(frame_count):
        patterns[i, :, :] = np.random.rand(height, width) > 0.5
    return patterns

def video_to_sci_encoder_3_channel(video_path, output_path, pattern_output_dir, frame_count_per_channel=10, start_time_msec=0):
    """Convert a video to a 3-channel SCI encoded snapshot, starting from start_time_msec, and save modulation patterns."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Ensure the output directory for patterns exists
    if not os.path.exists(pattern_output_dir):
        os.makedirs(pattern_output_dir)
    
    # Set the start position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_frame_count = frame_count_per_channel * 3  # 3 channels (R, G, B)
    
    # Generate modulation patterns
    patterns = generate_modulation_patterns(total_frame_count, height, width)

    # Save each modulation pattern as an image
    for i, pattern in enumerate(patterns):
        pattern_image = (pattern * 255).astype(np.uint8)  # Convert pattern to 8-bit image
        cv2.imwrite(os.path.join(pattern_output_dir, f'pattern_{i:02d}.png'), pattern_image)

    # Prepare empty frames for the SCI snapshot for each channel
    sci_snapshot_channels = [np.zeros((height, width), dtype=np.float32) for _ in range(3)]

    # Process video frames for each channel
    for channel_index in range(3):  # 0: Red, 1: Green, 2: Blue
        for i in range(frame_count_per_channel):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video.")
                return

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

            # Select the correct pattern for this frame
            pattern_index = channel_index * frame_count_per_channel + i
            pattern = patterns[pattern_index]

            # Modulate the frame
            modulated_frame = gray_frame * pattern

            # Add the modulated frame to the SCI snapshot for the current channel
            sci_snapshot_channels[channel_index] += modulated_frame

    # Normalize the SCI snapshots for each channel and combine them
    for i in range(3):
        sci_snapshot_channels[i] = sci_snapshot_channels[i] / frame_count_per_channel * 255
        sci_snapshot_channels[i] = sci_snapshot_channels[i].astype(np.uint8)

    # Stack the channels to create a 3-channel image
    sci_snapshot = np.stack(sci_snapshot_channels, axis=-1)

    # Save the 3-channel SCI snapshot
    cv2.imwrite(output_path, sci_snapshot)
    
    # Release the video capture object
    cap.release()
    print(f"3-channel SCI snapshot saved to {output_path}")
    print(f"Modulation patterns saved to {pattern_output_dir}")

# Example usage
video_path = "input/video12fps.mp4"
output_path = "output/output.png"
pattern_output_dir = "output/patterns"
start_time_msec = 90000  # Start at 10 seconds into the video
frame_count_per_channel = 10
video_to_sci_encoder_3_channel(video_path, output_path, pattern_output_dir, frame_count_per_channel, start_time_msec=start_time_msec)
