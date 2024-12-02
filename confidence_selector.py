import os
import json
import shutil
import argparse

def select_low_confidence_images(checkpoint_file, threshold, output_dir, move=False):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the checkpoint file
    try:
        with open(checkpoint_file, 'r') as f:
            confidence_scores = json.load(f)
    except Exception as e:
        print(f"Error reading checkpoint file: {e}")
        return
    
    # Filter and process low-confidence images
    low_confidence_images = [
        (image_path, confidence) 
        for image_path, confidence in confidence_scores 
        if confidence <= threshold
    ]
    
    # Sort by confidence to see the range
    low_confidence_images.sort(key=lambda x: x[1])
    
    # Copy or move images
    for image_path, confidence in low_confidence_images:
        try:
            # Generate destination filename with confidence score
            filename = os.path.basename(image_path)
            filename_base, filename_ext = os.path.splitext(filename)
            new_filename = f"{filename_base}_conf_{confidence:.4f}{filename_ext}"
            
            # Construct destination path
            dest_path = os.path.join(output_dir, new_filename)
            
            # Copy or move the image based on the move flag
            if move:
                shutil.move(image_path, dest_path)
                print(f"Moved: {image_path} to {dest_path}")
            else:
                shutil.copy2(image_path, dest_path)
                print(f"Copied: {image_path} to {dest_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Print summary
    print(f"Found {len(low_confidence_images)} images below threshold {threshold}")
    print("Confidence range:")
    if low_confidence_images:
        print(f"  Lowest:  {low_confidence_images[0][1]:.4f}")
        print(f"  Highest: {low_confidence_images[-1][1]:.4f}")
    
    return low_confidence_images

def query_image_confidence(checkpoint_file, image_name):
    """
    Query the confidence value for a specific image by its filename.
    
    Args:
        checkpoint_file (str): Path to the checkpoint JSON file
        image_name (str): Name of the image to query
    
    Returns:
        float or None: Confidence value of the image, or None if not found
    """
    try:
        # Read the checkpoint file
        with open(checkpoint_file, 'r') as f:
            confidence_scores = json.load(f)
        
        # Look for the image by its filename
        for image_path, confidence in confidence_scores:
            # Check if the image name matches (with or without full path)
            if os.path.basename(image_path) == image_name:
                print(f"Confidence for {image_name}: {confidence:.4f}")
                return confidence
        
        # If no matching image is found
        print(f"No confidence score found for image: {image_name}")
        return None
    
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {checkpoint_file}")
        return None
    except Exception as e:
        print(f"Unexpected error querying image confidence: {e}")
        return None

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Select low-confidence images or query image confidence')
    parser.add_argument('checkpoint_file', type=str, 
                        help='Path to the checkpoint JSON file')
    parser.add_argument('image_name', nargs='?', default=None,
                        help='Name of the image to query (optional)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for processing images (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='low_confidence_images', 
                        help='Output directory for low-confidence images')
    parser.add_argument('--move', action='store_true', 
                        help='Move images instead of copying them')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine the action based on arguments
    if args.image_name:
        # Query mode
        query_image_confidence(args.checkpoint_file, args.image_name)
    else:
        # Process mode
        select_low_confidence_images(
            args.checkpoint_file, 
            args.threshold, 
            args.output_dir,
            args.move
        )

if __name__ == "__main__":
    main()
    # Basic usage
    # python confidence_selector.py n_L_checkpoint.json

    # Specify a different threshold
    # python confidence_selector.py n_L_checkpoint.json --threshold 0.3

    # Specify a custom output directory
    # python confidence_selector.py F_U_checkpoint.json --threshold 0.0001 --output_dir bad_images_F_U --move
    # python confidence_selector.py query F_U_checkpoint.json F_U_3325_conf_0.0000.png