import os
import sys
import argparse
from google import genai
from google.genai import types

def generate_image(prompt, output_path, model="gemini-2.0-flash", aspect_ratio="1:1"):
    """
    Generates an image using the Google GenAI SDK (Nano Banana).
    Note: As of early 2026, image generation is integrated into the Flash models or 
    specific image models depending on the exact API version.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    print(f"Generating image with prompt: '{prompt}'...")
    
    try:
        # Using the standard image generation method for Gemini 3.1 / Nano Banana
        # The model name might need adjustment based on the latest available preview.
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            )
        )

        if not response.generated_images:
            print("Error: No images were generated.")
            sys.exit(1)

        for i, generated_image in enumerate(response.generated_images):
            # Save the first image to the specified output path
            generated_image.image.save(output_path)
            print(f"Successfully saved image to: {output_path}")
            break

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Nano Banana (Gemini).")
    parser.add_argument("prompt", help="The text prompt for image generation.")
    parser.add_argument("--output", "-o", default="output.png", help="Path to save the generated image.")
    parser.add_argument("--model", "-m", default="imagen-3.0-generate-002", help="The model to use (default: imagen-3.0-generate-002).")
    parser.add_argument("--ratio", "-r", default="1:1", choices=["1:1", "4:3", "3:4", "16:9", "9:16"], help="Aspect ratio.")

    args = parser.parse_args()
    
    generate_image(args.prompt, args.output, args.model, args.ratio)
