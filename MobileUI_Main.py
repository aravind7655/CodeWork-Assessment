import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionPipeline

# Initialize models
def initialize_models():
    """Initialize GPT-2 and Stable Diffusion models with error handling."""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        print("Using", device, "for image generation.")
        return tokenizer, model, pipe

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Generate a prompt using GPT-2
def generate_prompt(user_input, tokenizer, model, max_length=50, num_return_sequences=1):
    """Generate a prompt based on user input using GPT-2."""
    try:
        prompt = f"Create a mobile UI design for: {user_input}"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        st.error(f"Error generating prompt: {e}")
        return None

# Validate user input
def validate_input(user_input):
    """Check if user input is valid (not empty)."""
    if not user_input or len(user_input.strip()) == 0:
        st.warning("Please enter a valid description.")
        return False
    return True

# Generate UI design image
def generate_ui_design(user_input, tokenizer, model, pipe, guidance_scale=7.5, num_inference_steps=50):
    """Generate a UI design image using Stable Diffusion."""
    if not validate_input(user_input):
        return None

    prompt = generate_prompt(user_input, tokenizer, model)
    st.write("Generated Prompt:", prompt)

    try:
        image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        image_path = "generated_ui_design.png"
        image.save(image_path)
        st.image(image, caption="Generated UI Design", use_column_width=True)
        st.success(f"Image saved as '{image_path}'")
        return image_path

    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Streamlit app
def main():
    """Main function to run the Mobile UI Design Generator."""
    st.title("Mobile UI Design Generator")
    st.write("Welcome to the Mobile UI Design Generator!")

    tokenizer, model, pipe = initialize_models()
    if tokenizer is None or model is None or pipe is None:
        st.error("Failed to initialize models. Exiting.")
        return

    user_input = st.text_input("Enter a description for the mobile UI design (or type 'exit' to quit):")

    if user_input:
        if user_input.lower() == 'exit':
            st.write("Exiting the application. Thank you for using the Mobile UI Design Generator!")
        else:
            result = generate_ui_design(user_input, tokenizer, model, pipe)
            if result is None:
                st.write("Failed to generate the design. Please try again with valid input.")

# Entry point of the program
if __name__ == "__main__":
    main()
