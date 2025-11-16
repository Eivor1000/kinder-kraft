"""
AI Story Generator using MPT-7B-StoryWriter-65k+ Model
A Streamlit application for generating creative stories using the Mosaic ML Story Writer model.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import os
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the MPT-7B-StoryWriter model and tokenizer with caching.
    Returns model, tokenizer, and device information.
    """
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine dtype based on device
        if device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            st.warning("⚠️ CUDA not available. Running on CPU (will be slower).")

        with st.spinner("Loading model and tokenizer... This may take a few minutes on first run."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                trust_remote_code=True
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                "mosaicml/mpt-7b-storywriter",
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )

            model.to(device)
            model.eval()

        return model, tokenizer, device

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("This might be due to insufficient memory or network issues. Please try again.")
        return None, None, None


def count_tokens(text, tokenizer):
    """Count the number of tokens in a text."""
    return len(tokenizer.encode(text))


def generate_story(model, tokenizer, device, prompt, max_tokens, temperature, top_p, top_k):
    """
    Generate a story based on the given prompt and parameters.
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_token_count = inputs.input_ids.shape[1]

        # Record start time
        start_time = time.time()

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate generation time
        generation_time = time.time() - start_time

        # Count output tokens
        output_token_count = outputs.shape[1]

        return {
            "text": generated_text,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "generation_time": generation_time
        }

    except Exception as e:
        st.error(f"Error generating story: {str(e)}")
        return None


def get_memory_usage():
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # Convert to GB


def create_download_link(text, filename):
    """Create a download button for the generated story."""
    return st.download_button(
        label="📥 Download Story as .txt",
        data=text,
        file_name=filename,
        mime="text/plain"
    )


# Example prompts
EXAMPLE_PROMPTS = {
    "Fantasy Adventure": "In a realm where magic flows through ancient forests, a young apprentice discovers a forgotten spellbook that holds the key to saving their kingdom from an encroaching darkness.",
    "Sci-Fi Mystery": "The year is 2157. Detective Sarah Chen receives a cryptic message from her future self, warning her about an event that will change humanity forever. She has 48 hours to prevent it.",
    "Historical Fiction": "On the eve of the Renaissance, a mysterious artist arrives in Florence with paintings that seem to capture not just images, but the very souls of their subjects.",
    "Contemporary Drama": "After inheriting her grandmother's bookshop in a quiet coastal town, Maya discovers a collection of letters that reveal a family secret spanning three generations.",
    "Horror Thriller": "The old lighthouse keeper's journal contained only one repeated entry: 'Don't look at the sea when the fog rolls in.' But tonight, curiosity got the better of me."
}


def main():
    """Main application function."""

    # Header
    st.title("📚 AI Story Generator - MPT-7B-StoryWriter")
    st.markdown("Generate creative stories using the powerful MPT-7B-StoryWriter-65k+ model")

    # Load model
    model, tokenizer, device = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error("Failed to load model. Please refresh the page to try again.")
        return

    # Display model info
    st.success(f"✅ Model loaded successfully on {device.upper()}")

    # Sidebar - Generation Parameters
    st.sidebar.header("⚙️ Generation Parameters")

    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum number of tokens to generate"
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Controls randomness. Lower = more focused, Higher = more creative"
    )

    top_p = st.sidebar.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Cumulative probability cutoff for token selection"
    )

    top_k = st.sidebar.slider(
        "Top-k",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Number of top tokens to consider. 0 = disabled"
    )

    # Memory usage display
    if st.sidebar.checkbox("Show Memory Usage"):
        mem_usage = get_memory_usage()
        st.sidebar.metric("Memory Usage", f"{mem_usage:.2f} GB")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app uses the MPT-7B-StoryWriter-65k+ model, "
        "which is designed for long-form creative writing and can handle "
        "contexts up to 65,000 tokens."
    )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Story Prompt")

        # Example prompts expander
        with st.expander("💡 Example Prompts - Click to use"):
            for category, prompt_text in EXAMPLE_PROMPTS.items():
                if st.button(category, key=f"example_{category}"):
                    st.session_state.prompt = prompt_text

        # Text area for prompt
        prompt = st.text_area(
            "Enter your story prompt:",
            value=st.session_state.get("prompt", ""),
            height=200,
            placeholder="Once upon a time in a faraway land...",
            key="prompt_input"
        )

        # Update session state
        if prompt:
            st.session_state.prompt = prompt
            prompt_tokens = count_tokens(prompt, tokenizer)
            st.caption(f"Prompt tokens: {prompt_tokens}")

        # Action buttons
        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            generate_button = st.button("🚀 Generate Story", type="primary", use_container_width=True)

        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.prompt = ""
                st.session_state.generated_story = None
                st.rerun()

    with col2:
        st.subheader("📖 Generated Story")

        # Generate story when button is clicked
        if generate_button:
            if not prompt or prompt.strip() == "":
                st.warning("Please enter a story prompt first!")
            else:
                with st.spinner("Generating your story... ✨"):
                    result = generate_story(
                        model, tokenizer, device,
                        prompt, max_tokens, temperature, top_p, top_k
                    )

                if result:
                    st.session_state.generated_story = result
                    st.rerun()

        # Display generated story
        if st.session_state.get("generated_story"):
            result = st.session_state.generated_story

            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Input Tokens", result["input_tokens"])
            with metric_col2:
                st.metric("Output Tokens", result["output_tokens"])
            with metric_col3:
                st.metric("Generation Time", f"{result['generation_time']:.2f}s")

            st.markdown("---")

            # Display story
            st.markdown(result["text"])

            st.markdown("---")

            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_story_{timestamp}.txt"
            create_download_link(result["text"], filename)
        else:
            st.info("👆 Enter a prompt and click 'Generate Story' to begin!")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by MPT-7B-StoryWriter-65k+ | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
