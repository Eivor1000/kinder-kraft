"""
AI Story Generator with Multiple Model Support via HuggingFace Inference API
A Streamlit application for generating creative stories using cloud-hosted models.
No local downloads required - uses HuggingFace Inference API.
"""

import streamlit as st
from huggingface_hub import InferenceClient
import time
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
        "max_context": 65536,
        "max_tokens_limit": 5000,
        "description": "Specialized for long-form creative writing with 65k token context window",
        "best_for": "Continuing long stories, detailed world-building, extended narratives",
        "default_max_tokens": 500
    },
    "GOAT-70B-Storytelling": {
        "name": "GOAT-AI/GOAT-70B-Storytelling",
        "max_context": 4096,
        "max_tokens_limit": 2000,
        "description": "70B parameter model specialized for novel writing and character-driven stories",
        "best_for": "Structured novel plots, character development, script writing",
        "default_max_tokens": 400
    }
}


def get_hf_token():
    """
    Get HuggingFace API token from session state or environment variable.
    """
    # Check session state first
    if 'hf_token' in st.session_state and st.session_state.hf_token:
        return st.session_state.hf_token

    # Check environment variable
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        st.session_state.hf_token = token
        return token

    return None


def initialize_client(api_token):
    """
    Initialize HuggingFace Inference Client.
    """
    try:
        client = InferenceClient(token=api_token)
        return client
    except Exception as e:
        st.error(f"Error initializing client: {str(e)}")
        return None


def count_tokens_estimate(text):
    """
    Estimate token count (roughly 4 characters per token).
    This is an approximation since we don't have the tokenizer locally.
    """
    return len(text) // 4


def generate_story(client, model_name, prompt, max_tokens, temperature, top_p, top_k, model_type):
    """
    Generate a story using HuggingFace Inference API.
    """
    try:
        # Estimate input tokens
        input_token_estimate = count_tokens_estimate(prompt)

        # Check approximate context length limits
        max_context = MODEL_CONFIGS[model_type]["max_context"]

        if input_token_estimate > max_context:
            st.error(f"⚠️ Input prompt (~{input_token_estimate} tokens) may exceed model's context limit ({max_context} tokens)")
            st.info("Please shorten your prompt and try again.")
            return None

        # Record start time
        start_time = time.time()

        # Prepare generation parameters - using correct API format
        generation_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
        }

        # Add top_k only if it's greater than 0
        if top_k > 0:
            generation_params["top_k"] = top_k

        # Add repetition penalty (model-specific)
        if model_type == "MPT-7B-StoryWriter":
            generation_params["repetition_penalty"] = 1.1
        else:  # GOAT model
            generation_params["repetition_penalty"] = 1.15

        # Show debug info
        with st.expander("🔍 Debug Info (click to expand)"):
            st.code(f"Model: {model_name}\nParameters: {generation_params}")

        # Generate using text_generation
        # Note: Some models return just the new text, others return full text
        response = client.text_generation(
            prompt,
            model=model_name,
            **generation_params
        )

        # Calculate generation time
        generation_time = time.time() - start_time

        # Estimate output tokens
        output_token_estimate = count_tokens_estimate(response)

        # Calculate tokens per second
        tokens_generated = output_token_estimate - input_token_estimate
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

        return {
            "text": response,
            "input_tokens": input_token_estimate,
            "output_tokens": output_token_estimate,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second
        }

    except Exception as e:
        error_msg = str(e)

        # Show full error for debugging
        st.error(f"**Error generating story:**")
        st.code(error_msg)

        # Handle common API errors with helpful messages
        if "authorization" in error_msg.lower() or "401" in error_msg:
            st.info("❌ **Authorization Error**\n\n"
                   "Your HuggingFace API token may be invalid or expired.\n\n"
                   "**Solutions:**\n"
                   "- Check your token at https://huggingface.co/settings/tokens\n"
                   "- Ensure it has 'read' permissions\n"
                   "- Generate a new token if needed\n"
                   "- Click 'Change Token' in the sidebar")
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            st.info("⏱️ **Rate Limit Exceeded**\n\n"
                   "You've made too many requests.\n\n"
                   "**Solutions:**\n"
                   "- Wait 5-10 minutes before trying again\n"
                   "- Reduce max_tokens to make shorter requests\n"
                   "- Consider upgrading to HuggingFace Pro")
        elif "model is currently loading" in error_msg.lower() or "loading" in error_msg.lower():
            st.info("🔄 **Model is Loading**\n\n"
                   "The model is warming up on HuggingFace servers.\n\n"
                   "**Solution:**\n"
                   "- Wait 30-60 seconds and try again\n"
                   "- The next request will be faster")
        elif "timeout" in error_msg.lower():
            st.info("⏰ **Request Timeout**\n\n"
                   "The request took too long.\n\n"
                   "**Solutions:**\n"
                   "- Reduce max_tokens parameter\n"
                   "- Try again in a moment\n"
                   "- Check your internet connection")
        elif "not found" in error_msg.lower() or "404" in error_msg:
            st.info("🔍 **Model Not Found**\n\n"
                   "The model may not be available via Inference API.\n\n"
                   "**Note:** Some models are not available through the free Inference API.\n"
                   "Try the other model or check HuggingFace model page.")
        else:
            st.info("💡 **Troubleshooting Tips:**\n\n"
                   "1. Check your internet connection\n"
                   "2. Try reducing max_tokens\n"
                   "3. Try the other model\n"
                   "4. Wait a moment and try again\n"
                   "5. Check HuggingFace status: https://status.huggingface.co")

        return None


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
    | **Hosting** | HuggingFace Cloud | HuggingFace Cloud |
    | **Recommended Max Tokens** | 500-2000 | 300-1000 |
    | **Specialty** | Ultra-long context | Novel-quality prose |
    """)


def main():
    """Main application function."""

    # Header
    st.title("📚 AI Story Generator - Cloud API")
    st.markdown("Generate creative stories using HuggingFace Inference API - No downloads required!")

    # API Token Input
    st.sidebar.header("🔑 API Configuration")

    # Check if token exists
    existing_token = get_hf_token()

    if existing_token:
        st.sidebar.success("✅ API Token configured")
        if st.sidebar.button("Change Token"):
            st.session_state.hf_token = None
            st.rerun()
        api_token = existing_token
    else:
        st.sidebar.info("Enter your HuggingFace API token to get started")
        api_token_input = st.sidebar.text_input(
            "HuggingFace API Token:",
            type="password",
            help="Get your token at https://huggingface.co/settings/tokens"
        )

        if api_token_input:
            st.session_state.hf_token = api_token_input
            api_token = api_token_input
            st.rerun()
        else:
            st.warning("⚠️ Please enter your HuggingFace API token in the sidebar to continue.")
            st.info("""
            **How to get your HuggingFace API token:**
            1. Go to https://huggingface.co/settings/tokens
            2. Click "New token"
            3. Give it a name and select "read" permissions
            4. Copy the token and paste it in the sidebar

            **Note:** The token is free and only takes a minute to create!
            """)

            with st.expander("📊 Model Information"):
                display_model_comparison()

            return

    st.sidebar.markdown("---")

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

    # Initialize client
    model_name = model_config["name"]
    client = initialize_client(api_token)

    if client is None:
        st.error("Failed to initialize API client. Please check your token and try again.")
        return

    # Display API status
    st.success(f"✅ Connected to HuggingFace API - Using {selected_model}")

    # Sidebar - Generation Parameters
    st.sidebar.header("⚙️ Generation Parameters")

    # Dynamic max tokens based on selected model
    max_tokens_limit = model_config["max_tokens_limit"]
    default_max_tokens = model_config["default_max_tokens"]

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
            - Ideal for serialized fiction
            """)
        else:
            st.markdown("""
            **GOAT-70B Tips:**
            - Superior for novel-quality prose
            - Excellent character development
            - Great for screenplay/script format
            - Better at following complex instructions
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
            prompt_tokens = count_tokens_estimate(prompt)

            # Color-code token count based on context limit
            max_context = model_config["max_context"]
            token_percentage = (prompt_tokens / max_context) * 100

            if token_percentage < 50:
                st.caption(f"✅ Estimated prompt tokens: ~{prompt_tokens} / {max_context:,} ({token_percentage:.1f}%)")
            elif token_percentage < 80:
                st.caption(f"⚠️ Estimated prompt tokens: ~{prompt_tokens} / {max_context:,} ({token_percentage:.1f}%)")
            else:
                st.caption(f"🔴 Estimated prompt tokens: ~{prompt_tokens} / {max_context:,} ({token_percentage:.1f}%)")
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
                        client, model_name,
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
                st.metric("Input Tokens", f"~{result['input_tokens']}")
            with metric_col2:
                st.metric("Output Tokens", f"~{result['output_tokens']}")
            with metric_col3:
                st.metric("Time", f"{result['generation_time']:.2f}s")
            with metric_col4:
                st.metric("Speed", f"~{result['tokens_per_second']:.1f} tok/s")

            # Show model used
            st.caption(f"Generated with: **{result.get('model', 'Unknown')}** via HuggingFace API")

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
            download_content += f"Via: HuggingFace Inference API\n"
            download_content += f"Timestamp: {result.get('timestamp', datetime.now())}\n"
            download_content += f"Generation time: {result['generation_time']:.2f}s\n"
            download_content += f"Tokens (estimated): ~{result['input_tokens']} → ~{result['output_tokens']}\n"
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
                          f"~{gen['output_tokens']} tokens in {gen['generation_time']:.1f}s")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        f"Powered by {selected_model} via HuggingFace Inference API | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
