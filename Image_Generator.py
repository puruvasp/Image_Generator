import torch
from diffusers import StableDiffusionPipeline


def generate_image(prompt):
    # Load the pre-trained Stable Diffusion model
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the image from the prompt
    with torch.autocast("cuda"):
        image = pipeline(prompt).images[0]

    # Save the image
    image_path = "generated_image.png"
    image.save(image_path)
    print(f"Image saved to {image_path}")


if __name__ == "__main__":
    user_prompt = input("Enter your prompt for the image: ")
    generate_image(user_prompt)
