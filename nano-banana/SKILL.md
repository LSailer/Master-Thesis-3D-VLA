---
name: nano-banana
description: Generates high-quality images using Google's Nano Banana models (Gemini Image Generation). Use when the user wants to create visuals, diagrams, or concept art based on text descriptions.
---

# Nano Banana Image Generation

This skill provides a tool to generate images using Google's state-of-the-art image generation models.

## Workflow

1.  **Formulate a detailed prompt**: Include specific details about style, lighting, composition, and key elements.
2.  **Generate the image**: Use the `scripts/generate_image.py` script.
3.  **Provide the output**: Show the user where the image was saved.

## Usage

Run the generator via `uv` or `python`:

```bash
uv run nano-banana/scripts/generate_image.py "your detailed prompt here" --output "path/to/image.png"
```

### Options
- `--output`, `-o`: Output file path (default: `output.png`).
- `--model`, `-m`: Model name (default: `imagen-3.0-generate-002`).
- `--ratio`, `-r`: Aspect ratio (`1:1`, `4:3`, `3:4`, `16:9`, `9:16`).

## Example Triggers
- "Generate an image for my thesis cover."
- "Create a visual of a robot in a 3D point cloud room."
- "Use nano-banana to make a 16:9 wallpaper of a cyberpunk city."

## Environment Variables
- `GOOGLE_API_KEY`: Required to authenticate with the Gemini API.
