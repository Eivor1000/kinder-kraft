"""
AI Story Generator with Multiple Model Support
A Streamlit application for generating creative stories using MPT-7B-StoryWriter and GOAT-70B-Storytelling models.
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


# Model configurations
MODEL_CONFIGS = {
    "MPT-7B-StoryWriter": {
        "name": "mosaicml/mpt-7b-storywriter",
        "tokenizer": "EleutherAI/gpt-neox-20b",
        "max_context": 65536,
        "max_tokens_limit": 5000,
        "description": "Specialized for long-form creative writing with 65k token context window",
        "best_for": "Continuing long stories, detailed world-building, extended narratives"
    },
    "GOAT-70B-Storytelling": {
        "name": "GOAT-AI/GOAT-70B-Storytelling",
        "tokenizer": "GOAT-AI/GOAT-70B-Storytelling",
        "max_context": 4096,
        "max_tokens_limit": 2000,
        "description": "70B parameter model specialized for novel writing and character-driven stories",
        "best_for": "Structured novel plots, character development, script writing"
    }
}


@st.cache_resource
def load_mpt_model():
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

        with st.spinner("Loading MPT-7B-StoryWriter model... This may take a few minutes."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIGS["MPT-7B-StoryWriter"]["tokenizer"],
                trust_remote_code=True
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIGS["MPT-7B-StoryWriter"]["name"],
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )

            model.to(device)
            model.eval()

        return model, tokenizer, device

    except Exception as e:
        st.error(f"Error loading MPT model: {str(e)}")
        st.info("This might be due to insufficient memory or network issues. Please try again.")
        return None, None, None


@st.cache_resource
def load_goat_model():
    """
    Load the GOAT-70B-Storytelling model and tokenizer with caching.
    Uses device_map="auto" to handle the large 70B model.
    Returns model, tokenizer, and device information.
    """
    try:
        if not torch.cuda.is_available():
            st.error("⚠️ GOAT-70B requires a GPU. CUDA is not available.")
            st.info("Please use MPT-7B-StoryWriter model instead, or ensure CUDA is properly configured.")
            return None, None, None

        with st.spinner("Loading GOAT-70B-Storytelling model... This may take several minutes."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIGS["GOAT-70B-Storytelling"]["tokenizer"],
                trust_remote_code=True
            )

            # Load model with automatic device mapping for large models
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIGS["GOAT-70B-Storytelling"]["name"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            model.eval()

        return model, tokenizer, "auto"

    except Exception as e:
        st.error(f"Error loading GOAT model: {str(e)}")
        if "out of memory" in str(e).lower():
            st.error("⚠️ Out of memory error. GOAT-70B requires significant GPU memory (typically 80GB+).")
            st.info("Try using MPT-7B-StoryWriter instead, which requires much less memory (~16GB).")
        else:
            st.info("This might be due to insufficient memory, network issues, or model availability. Please try again.")
        return None, None, None


def count_tokens(text, tokenizer):
    """Count the number of tokens in a text."""
    return len(tokenizer.encode(text))


def generate_story(model, tokenizer, device, prompt, max_tokens, temperature, top_p, top_k, model_type):
    """
    Generate a story based on the given prompt and parameters.
    Handles both MPT and GOAT models appropriately.
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to device only if not using auto device mapping
        if device != "auto":
            inputs = inputs.to(device)
        else:
            # For device_map="auto", move to model's first device
            inputs = inputs.to(model.device)

        input_token_count = inputs.input_ids.shape[1]

        # Check context length limits
        max_context = MODEL_CONFIGS[model_type]["max_context"]
        if input_token_count > max_context:
            st.error(f"⚠️ Input prompt ({input_token_count} tokens) exceeds model's context limit ({max_context} tokens)")
            st.info("Please shorten your prompt and try again.")
            return None

        # Record start time
        start_time = time.time()

        # Generate with model-specific parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Add top_k only if it's greater than 0
        if top_k > 0:
            generation_kwargs["top_k"] = top_k

        # Add repetition penalty (model-specific)
        if model_type == "MPT-7B-StoryWriter":
            generation_kwargs["repetition_penalty"] = 1.1
        else:  # GOAT model
            generation_kwargs["repetition_penalty"] = 1.15

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                **generation_kwargs
            )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate generation time
        generation_time = time.time() - start_time

        # Count output tokens
        output_token_count = outputs.shape[1]

        # Calculate tokens per second
        tokens_per_second = (output_token_count - input_token_count) / generation_time if generation_time > 0 else 0

        return {
            "text": generated_text,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second
        }

    except torch.cuda.OutOfMemoryError:
        st.error("⚠️ GPU out of memory!")
        st.info("Try reducing the max tokens parameter or using a smaller model.")
        return None
    except Exception as e:
        st.error(f"Error generating story: {str(e)}")
        return None


def get_memory_usage():
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # Convert to GB


def get_gpu_memory():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB
        return allocated, reserved
    return None, None


def create_download_link(text, filename):
    """Create a download button for the generated story."""
    return st.download_button(
        label="📥 Download Story as .txt",
        data=text,
        file_name=filename,
        mime="text/plain"
    )


# Example prompts for MPT model (long-form content)
MPT_EXAMPLE_PROMPTS = {
    "Epic Fantasy Continuation": "The Chronicles of Eldoria, Chapter 47:\n\nAfter three long years of journeying through the Forbidden Wastes, Aria finally stood at the gates of the Crystal Citadel. The ancient prophecy had led her here, but nothing could have prepared her for what she was about to discover...",
    "Sci-Fi Space Opera": "Captain's Log, Stardate 7734.2:\n\nWe've been tracking the mysterious signal for six weeks now, deeper into uncharted space than any Federation vessel has ever ventured. The crew is getting restless, but I have a feeling we're on the verge of the most important discovery in human history...",
    "Historical Epic": "Constantinople, 1453:\n\nAs dawn broke over the ancient city, Emperor Constantine XI knew this would be the final day. The Ottoman forces had surrounded the walls, and the sound of their war drums echoed through the streets. But within the Hagia Sophia, a secret was about to be revealed that would change everything...",
    "Mystery Noir": "The rain hadn't stopped for three days when she walked into my office. The dame had trouble written all over her, the kind of trouble that usually ended with someone in a pine box. But her story about the missing heirloom was just the beginning of something much darker..."
}

# Example prompts for GOAT model (structured narratives)
GOAT_EXAMPLE_PROMPTS = {
    "Character-Driven Drama": "Write a story about Emma, a talented but struggling violinist who discovers her late grandmother's diary revealing she was once a famous musician who mysteriously gave up her career. Explore Emma's journey as she uncovers family secrets and finds her own voice.",
    "Romance Novel Opening": "Create the opening chapter of a romance novel where two rival architects, Maya and James, are forced to collaborate on the city's most important project. Their professional competition masks a deeper connection neither is willing to admit.",
    "Psychological Thriller": "Develop a story about Dr. Sarah Mitchell, a psychiatrist who starts having vivid dreams about her patients' sessions—only to discover the dreams are revealing crimes her patients haven't confessed to yet.",
    "Coming-of-Age Story": "Tell the story of Marcus, a 16-year-old from a small town who dreams of becoming a chef. When he gets an internship at a prestigious restaurant in the city, he must navigate a world completely different from anything he's known.",
    "Script Format - Scene": "Write a screenplay scene: INT. DETECTIVE'S OFFICE - NIGHT. Two detectives interrogate a suspect who claims to be from the future. The suspect knows intimate details about the detectives' lives that haven't happened yet."
}


def display_model_comparison():
    """Display a comparison table of available models."""
    st.markdown("""
    ### Model Comparison

    | Feature | MPT-7B-StoryWriter | GOAT-70B-Storytelling |
    |---------|-------------------|----------------------|
    | **Parameters** | 7 Billion | 70 Billion |
    | **Context Window** | 65,536 tokens | 4,096 tokens |
    | **Best For** | Long-form content, extended narratives | Structured plots, character development |
    | **Speed** | Faster | Slower |
    | **Memory Required** | ~16GB GPU | ~80GB GPU |
    | **Recommended Max Tokens** | 500-2000 | 300-1000 |
    | **Specialty** | Ultra-long context | Novel-quality prose |
    """)


def main():
    """Main application function."""

    # Header
    st.title("📚 AI Story Generator - Multi-Model")
    st.markdown("Generate creative stories using state-of-the-art language models")

    # Sidebar - Model Selection
    st.sidebar.header("🤖 Model Selection")

    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        options=list(MODEL_CONFIGS.keys()),
        format_func=lambda x: f"{x} ({MODEL_CONFIGS[x]['max_context']//1000}k context)",
        help="Select the AI model to use for story generation"
    )

    # Display model information
    model_config = MODEL_CONFIGS[selected_model]
    st.sidebar.info(
        f"**{selected_model}**\n\n"
        f"{model_config['description']}\n\n"
        f"**Best for:** {model_config['best_for']}"
    )

    # Context window information
    st.sidebar.metric(
        "Max Context Window",
        f"{model_config['max_context']:,} tokens"
    )

    st.sidebar.markdown("---")

    # Load the selected model
    if selected_model == "MPT-7B-StoryWriter":
        model, tokenizer, device = load_mpt_model()
    else:  # GOAT-70B-Storytelling
        model, tokenizer, device = load_goat_model()

    if model is None or tokenizer is None:
        st.error("Failed to load model. Please try selecting a different model or refresh the page.")

        # Show model comparison to help user choose alternative
        with st.expander("📊 Compare Models"):
            display_model_comparison()

        return

    # Display model info
    device_display = device.upper() if device != "auto" else "AUTO (Multi-GPU)"
    st.success(f"✅ {selected_model} loaded successfully on {device_display}")

    # Sidebar - Generation Parameters
    st.sidebar.header("⚙️ Generation Parameters")

    # Dynamic max tokens based on selected model
    max_tokens_limit = model_config["max_tokens_limit"]
    default_max_tokens = 500 if selected_model == "MPT-7B-StoryWriter" else 400

    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=max_tokens_limit,
        value=default_max_tokens,
        step=50,
        help=f"Maximum number of tokens to generate (limit: {max_tokens_limit})"
    )

    # Warning for GOAT model with high token count
    if selected_model == "GOAT-70B-Storytelling" and max_tokens > 1500:
        st.sidebar.warning("⚠️ High token count may be slow with GOAT-70B. Consider 300-1000 tokens for optimal performance.")

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
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show Memory Usage"):
        mem_usage = get_memory_usage()
        st.sidebar.metric("System Memory", f"{mem_usage:.2f} GB")

        gpu_allocated, gpu_reserved = get_gpu_memory()
        if gpu_allocated is not None:
            col1, col2 = st.sidebar.columns(2)
            col1.metric("GPU Allocated", f"{gpu_allocated:.2f} GB")
            col2.metric("GPU Reserved", f"{gpu_reserved:.2f} GB")

    # Model recommendations
    st.sidebar.markdown("---")
    with st.sidebar.expander("💡 Model Recommendations"):
        if selected_model == "MPT-7B-StoryWriter":
            st.markdown("""
            **MPT-7B Tips:**
            - Excellent for continuing existing long stories
            - Can maintain context over tens of thousands of words
            - Works well with detailed prompts
            - Faster generation speed
            - Lower memory requirements
            """)
        else:
            st.markdown("""
            **GOAT-70B Tips:**
            - Superior for novel-quality prose
            - Excellent character development
            - Great for screenplay/script format
            - Better at following complex instructions
            - Requires powerful GPU (80GB+ recommended)
            - Keep prompts focused and structured
            """)

    # Model comparison table
    with st.sidebar.expander("📊 Compare Models"):
        display_model_comparison()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Story Prompt")

        # Model-specific example prompts
        example_prompts = MPT_EXAMPLE_PROMPTS if selected_model == "MPT-7B-StoryWriter" else GOAT_EXAMPLE_PROMPTS

        with st.expander(f"💡 Example Prompts for {selected_model} - Click to use"):
            for category, prompt_text in example_prompts.items():
                if st.button(category, key=f"example_{category}"):
                    st.session_state.prompt = prompt_text
                    st.rerun()

        # Text area for prompt
        prompt = st.text_area(
            "Enter your story prompt:",
            value=st.session_state.get("prompt", ""),
            height=250,
            placeholder="Enter your story idea or beginning...",
            key="prompt_input"
        )

        # Update session state and show token count
        if prompt:
            st.session_state.prompt = prompt
            prompt_tokens = count_tokens(prompt, tokenizer)

            # Color-code token count based on context limit
            max_context = model_config["max_context"]
            token_percentage = (prompt_tokens / max_context) * 100

            if token_percentage < 50:
                st.caption(f"✅ Prompt tokens: {prompt_tokens} / {max_context:,} ({token_percentage:.1f}%)")
            elif token_percentage < 80:
                st.caption(f"⚠️ Prompt tokens: {prompt_tokens} / {max_context:,} ({token_percentage:.1f}%)")
            else:
                st.caption(f"🔴 Prompt tokens: {prompt_tokens} / {max_context:,} ({token_percentage:.1f}%)")
                st.warning("Prompt is very long. Consider shortening for better results.")

        # Action buttons
        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            generate_button = st.button("🚀 Generate Story", type="primary", use_container_width=True)

        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.prompt = ""
                st.session_state.generated_story = None
                if 'generation_history' in st.session_state:
                    st.session_state.generation_history = []
                st.rerun()

    with col2:
        st.subheader("📖 Generated Story")

        # Generate story when button is clicked
        if generate_button:
            if not prompt or prompt.strip() == "":
                st.warning("Please enter a story prompt first!")
            else:
                with st.spinner(f"Generating your story with {selected_model}... ✨"):
                    result = generate_story(
                        model, tokenizer, device,
                        prompt, max_tokens, temperature, top_p, top_k,
                        selected_model
                    )

                if result:
                    result["model"] = selected_model
                    result["timestamp"] = datetime.now()
                    st.session_state.generated_story = result

                    # Add to history
                    if 'generation_history' not in st.session_state:
                        st.session_state.generation_history = []
                    st.session_state.generation_history.append(result)

                    st.rerun()

        # Display generated story
        if st.session_state.get("generated_story"):
            result = st.session_state.generated_story

            # Display metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Input Tokens", result["input_tokens"])
            with metric_col2:
                st.metric("Output Tokens", result["output_tokens"])
            with metric_col3:
                st.metric("Time", f"{result['generation_time']:.2f}s")
            with metric_col4:
                st.metric("Speed", f"{result['tokens_per_second']:.1f} tok/s")

            # Show model used
            st.caption(f"Generated with: **{result.get('model', 'Unknown')}**")

            st.markdown("---")

            # Display story
            st.markdown(result["text"])

            st.markdown("---")

            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_abbrev = "mpt" if selected_model == "MPT-7B-StoryWriter" else "goat"
            filename = f"story_{model_abbrev}_{timestamp}.txt"

            # Add metadata to download
            download_content = f"Generated with: {result.get('model', 'Unknown')}\n"
            download_content += f"Timestamp: {result.get('timestamp', datetime.now())}\n"
            download_content += f"Generation time: {result['generation_time']:.2f}s\n"
            download_content += f"Tokens: {result['input_tokens']} → {result['output_tokens']}\n"
            download_content += "\n" + "="*50 + "\n\n"
            download_content += result["text"]

            create_download_link(download_content, filename)
        else:
            st.info("👆 Enter a prompt and click 'Generate Story' to begin!")

    # Generation history
    if st.session_state.get('generation_history') and len(st.session_state.generation_history) > 1:
        with st.expander(f"📜 Generation History ({len(st.session_state.generation_history)} stories)"):
            for idx, gen in enumerate(reversed(st.session_state.generation_history), 1):
                st.markdown(f"**{idx}. {gen.get('model', 'Unknown')}** - "
                          f"{gen.get('timestamp', 'Unknown').strftime('%H:%M:%S')} - "
                          f"{gen['output_tokens']} tokens in {gen['generation_time']:.1f}s")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        f"Powered by {selected_model} | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
