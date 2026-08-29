import os
import sys
import json
import argparse
import base64

# Add parent directory to sys.path so imports work smoothly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.ml.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="AI Image Quality & Defect Detection CLI")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument("--save-heatmap", help="Path to save the output heatmap image (optional)", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {args.image_path} ...")
    result = run_inference(args.image_path)

    # If user requested to save the heatmap image
    if args.save_heatmap and "heatmap" in result and result["heatmap"]:
        try:
            header, encoded = result["heatmap"].split(",", 1)
            img_data = base64.b64decode(encoded)
            with open(args.save_heatmap, "wb") as f:
                f.write(img_data)
            print(f"Heatmap saved to: {args.save_heatmap}")
        except Exception as e:
            print(f"Warning: Could not save heatmap: {e}")

    # Print clean formatted JSON output (without long base64 string for clarity)
    display_result = {
        "quality_score": result["quality_score"],
        "quality_label": result["quality_label"],
        "issues": result["issues"],
        "stats": result["stats"]
    }
    print("\n--- Analysis Result ---")
    print(json.dumps(display_result, indent=2))

if __name__ == "__main__":
    main()
