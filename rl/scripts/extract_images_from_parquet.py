import pandas as pd
import os
import json
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import random

def decode_and_save_image(args):
    """
    Decode and save a single image.
    Returns: (original_index, success, file_path, file_name, error_message)
    """
    original_idx, image_data, save_dir, file_name = args

    try:
        # Use provided filename (preserves original name if it exists)
        file_path = os.path.join(save_dir, file_name)

        # Decode image based on type
        if isinstance(image_data, bytes):
            # Direct bytes
            img = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            # Base64 encoded string
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',', 1)[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
        else:
            # Assume it's already a PIL Image or similar
            img = image_data

        # Get image size before conversion
        image_size = img.size  # (width, height)

        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if img.mode in ("RGBA", "LA", "P"):
            # To uniformly handle all transparency cases, first convert image to RGBA mode.
            # If image is in 'P' mode with transparency, conversion will result in correct RGBA image.
            img = img.convert("RGBA")

            # Create a white background base image
            background = Image.new("RGB", img.size, (255, 255, 255))

            # Paste original image onto the background.
            # At this point img is already in RGBA mode, so it can serve as its own mask.
            # Pillow will automatically use its Alpha channel.
            background.paste(img, (0, 0), img)
            img = background  # Now img is the merged RGB image
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save with original format (PIL infers from file extension)
        img.save(file_path, quality=95)

        return (original_idx, True, file_path, file_name, image_size, None)

    except Exception as e:
        return (original_idx, False, None, None, None, str(e))


def extract_images_from_parquet(parquet_path, image_save_path, max_workers=16):
    """
    Extract images from parquet file and save them to disk.
    Updates the extra_info column with file paths.

    Args:
        parquet_path: Path to input parquet file
        image_save_path: Directory to save images
        max_workers: Number of threads for parallel processing
    """

    print(f"Loading parquet from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Create save directory if it doesn't exist
    os.makedirs(image_save_path, exist_ok=True)
    print(f"Saving images to: {image_save_path}")

    # Check if images column exists
    if 'images' not in df.columns:
        raise ValueError("No 'images' column found in the parquet file")

    # Prepare tasks for thread pool with original indices
    # Important: preserve existing image_file_name if it exists in extra_info
    # to maintain consistency with references in prompts
    tasks = []
    for idx in df.index:
        images = df.loc[idx, 'images']

        # Handle different image formats (single image or list of images)
        if isinstance(images, list):
            # Multiple images - we'll save the first one for now
            if len(images) > 0:
                image_data = images[0]
                tasks.append((idx, image_data))
        elif images is not None:
            # Single image
            tasks.append((idx, images))

    # Build final task list - use existing filename if available, otherwise use row index
    print(f"\nPreparing to process {len(tasks)} images...")
    final_tasks = []
    for i, (original_idx, image_data) in enumerate(tasks):
        # Check if there's an existing image_file_name in extra_info
        extra_info = df.loc[original_idx, 'extra_info']
        existing_filename = None

        if pd.notna(extra_info):
            if isinstance(extra_info, dict):
                existing_filename = extra_info.get('image_file_name')
            elif isinstance(extra_info, str):
                try:
                    parsed_info = json.loads(extra_info)
                    existing_filename = parsed_info.get('image_file_name')
                except:
                    pass

        # Use existing filename from extra_info (every image has a unique name)
        assert existing_filename is not None, f"Missing image_file_name in extra_info at index {original_idx}"
        file_name = existing_filename

        final_tasks.append((original_idx, image_data, image_save_path, file_name))

    print(f"Processing {len(final_tasks)} images with {max_workers} workers...")

    # Process images in parallel
    results = {}
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(decode_and_save_image, task): task for task in final_tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving images"):
            original_idx, success, file_path, file_name, image_size, error = future.result()

            if success:
                results[original_idx] = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'image_size': image_size
                }
            else:
                failed.append((original_idx, error))

    # Update extra_info column - build list of updated values
    print("\nUpdating extra_info column...")
    updated_extra_info = []

    for idx in tqdm(df.index, desc="Updating dataframe"):
        if idx in results:
            # Get existing extra_info or create new dict
            current_extra_info = df.loc[idx, 'extra_info']

            if pd.isna(current_extra_info):
                extra_info = {}
            else:
                if isinstance(current_extra_info, str):
                    try:
                        extra_info = json.loads(current_extra_info)
                    except:
                        extra_info = {}
                elif isinstance(current_extra_info, dict):
                    extra_info = current_extra_info.copy()
                else:
                    extra_info = {}

            # Add image file information
            extra_info['image_file_path'] = results[idx]['file_path']
            extra_info['image_file_name'] = results[idx]['file_name']
            extra_info['image_size'] = results[idx]['image_size']  # (width, height)

            updated_extra_info.append(extra_info)
        else:
            # Keep original value if no image was processed
            updated_extra_info.append(df.loc[idx, 'extra_info'])

    # Assign the entire column at once
    df['extra_info'] = updated_extra_info

    # Print statistics
    print(f"\n--- Statistics ---")
    print(f"Total images processed: {len(tasks)}")
    print(f"Successfully saved: {len(results)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFirst 10 failures:")
        for idx, error in failed[:10]:
            print(f"  Index {idx}: {error}")

    # Save updated parquet
    output_path = parquet_path.replace('.parquet', '_with_image_paths.parquet')
    print(f"\nSaving updated parquet to: {output_path}")
    df.to_parquet(output_path, index=False)
    print("Done!")

    return df, results, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract images from parquet and save to disk')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='Path to input parquet file')
    parser.add_argument('--image_save_path', type=str, required=True,
                        help='Directory to save extracted images')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Number of threads for parallel processing (default: 16)')

    args = parser.parse_args()

    extract_images_from_parquet(
        parquet_path=args.parquet_path,
        image_save_path=args.image_save_path,
        max_workers=args.max_workers
    )
