# AI Story Generator - Multi-Model Support

A powerful Streamlit application for generating creative stories using state-of-the-art language models. Choose between MPT-7B-StoryWriter for long-form narratives or GOAT-70B-Storytelling for novel-quality prose.

## Features

### Multi-Model Support
- **MPT-7B-StoryWriter**: 65k token context window, perfect for ultra-long stories
- **GOAT-70B-Storytelling**: 70B parameters, specialized for novel writing and character development
- Easy model switching with one click
- Model comparison table to help choose the right model
- Model-specific example prompts and recommendations

### Core Features
- **Customizable Parameters**: Fine-tune generation with temperature, top-p, top-k, and max tokens controls
- **Dynamic UI**: Interface adapts based on selected model
- **Token Tracking**: Monitor input and output token counts with context usage indicators
- **Performance Metrics**: View generation time, tokens per second, and memory usage
- **Export Stories**: Download generated stories as .txt files with metadata
- **Generation History**: Track multiple generations in a session
- **GPU Support**: Automatic detection and use of CUDA-enabled GPUs
- **Smart Device Mapping**: Automatic multi-GPU support for large models
- **User-Friendly Interface**: Clean, intuitive UI built with Streamlit

## Model Comparison

| Feature | MPT-7B-StoryWriter | GOAT-70B-Storytelling |
|---------|-------------------|----------------------|
| **Parameters** | 7 Billion | 70 Billion |
| **Context Window** | 65,536 tokens | 4,096 tokens |
| **Best For** | Long-form content, extended narratives | Structured plots, character development |
| **Speed** | Faster | Slower |
| **Memory Required** | ~16GB GPU | ~80GB GPU |
| **Recommended Max Tokens** | 500-2000 | 300-1000 |
| **Specialty** | Ultra-long context | Novel-quality prose |

## Requirements

### Minimum Requirements
- Python 3.8 or higher
- 16GB RAM (for MPT model)
- ~15GB disk space for model downloads

### Recommended for GOAT-70B
- Python 3.10 or higher
- CUDA-compatible GPU with 80GB+ VRAM (A100, H100)
- 32GB+ System RAM
- ~140GB disk space

### Recommended for MPT-7B
- CUDA-compatible GPU with 16GB+ VRAM
- 32GB System RAM

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd kinder-kraft
```

### 2. Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first time you select a model, it will be downloaded automatically. Download sizes:
- MPT-7B-StoryWriter: ~15GB
- GOAT-70B-Storytelling: ~140GB

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the App

1. **Select a Model**: Choose between MPT-7B-StoryWriter or GOAT-70B-Storytelling from the sidebar
2. **Review Model Info**: Check the model description, context window, and recommendations
3. **Enter a Story Prompt**: Type your story idea or click on model-specific example prompts
4. **Adjust Parameters**:
   - **Max Tokens**: Control the length (dynamically adjusted per model)
   - **Temperature**: Adjust creativity (0.1 = focused, 2.0 = very creative)
   - **Top-p**: Control diversity via nucleus sampling (0.1-1.0)
   - **Top-k**: Limit vocabulary per step (0-100, 0 = disabled)
5. **Generate Story**: Click the "Generate Story" button
6. **Review Metrics**: Check generation time, token count, and speed
7. **Download**: Export your story with metadata as a .txt file
8. **View History**: See all stories generated in the current session

### Model-Specific Example Prompts

**MPT-7B-StoryWriter** (Long-form content):
- Epic Fantasy Continuation
- Sci-Fi Space Opera
- Historical Epic
- Mystery Noir

**GOAT-70B-Storytelling** (Structured narratives):
- Character-Driven Drama
- Romance Novel Opening
- Psychological Thriller
- Coming-of-Age Story
- Script Format Scenes

## Configuration

### GPU vs CPU

The app automatically detects CUDA availability:
- **MPT Model with GPU**: Uses bfloat16 precision
- **MPT Model without GPU**: Falls back to CPU with float32 (slower but functional)
- **GOAT Model**: Requires GPU (will show error without CUDA)

### Device Mapping

For the GOAT-70B model, the app uses `device_map="auto"` which:
- Automatically distributes the model across available GPUs
- Optimizes memory usage
- Handles models larger than single GPU memory

### Memory Optimization

If you encounter out-of-memory errors:
- **For MPT-7B**: Reduce max_tokens, use CPU, or close other applications
- **For GOAT-70B**: Requires high-end GPU (80GB+), consider using MPT instead
- Enable "Show Memory Usage" to monitor RAM and GPU usage

## Technical Details

### MPT-7B-StoryWriter

- **Model**: [mosaicml/mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- **Tokenizer**: [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)
- **Context Length**: Up to 65,536 tokens
- **Parameters**: 7 billion
- **Precision**: bfloat16 (GPU) or float32 (CPU)
- **Special Features**: FlashAttention support, ultra-long context

### GOAT-70B-Storytelling

- **Model**: [GOAT-AI/GOAT-70B-Storytelling](https://huggingface.co/GOAT-AI/GOAT-70B-Storytelling)
- **Tokenizer**: Built-in tokenizer
- **Context Length**: 4,096 tokens
- **Parameters**: 70 billion
- **Precision**: bfloat16
- **Special Features**: Novel-quality prose, character development

### Key Features in Code

- **Separate Model Loaders**: `load_mpt_model()` and `load_goat_model()` with `@st.cache_resource`
- **Unified Generation**: Single `generate_story()` function handles both models
- **Dynamic UI**: Parameters adjust automatically based on selected model
- **Error Handling**: Specific error messages for OOM, missing CUDA, etc.
- **Session State**: Maintains prompts, stories, and generation history
- **Model-Specific Prompts**: Different example prompts tailored to each model's strengths

## Troubleshooting

### Model Selection Issues

**Issue**: GOAT model won't load
**Solution**:
- Verify CUDA is available: `torch.cuda.is_available()`
- Check GPU memory: Need 80GB+ for GOAT-70B
- Try MPT-7B-StoryWriter instead (much lower requirements)

### CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`
**Solution**:
- **MPT Model**: Try reducing max_tokens or switching to CPU
- **GOAT Model**: Requires high-end GPU (A100 80GB or H100)
- Close other GPU-intensive applications
- Use the "Show Memory Usage" feature to monitor usage

### Slow Generation

**Issue**: Story generation takes a long time
**Solution**:
- **MPT on CPU**: Expected; consider using GPU or reduce max_tokens
- **GOAT Model**: Expected due to 70B parameters; use 300-1000 max tokens
- Check tokens/second metric to see actual speed
- Ensure GPU is being used (check success message)

### Import Errors

**Issue**: `ModuleNotFoundError`
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Model Download Issues

**Issue**: Download interrupted or fails
**Solution**:
- Check internet connection and disk space
- For GOAT: Ensure you have ~140GB free
- Clear HuggingFace cache if needed: `rm -rf ~/.cache/huggingface/`
- Download may take hours for GOAT-70B

### Context Length Exceeded

**Issue**: Prompt too long for model
**Solution**:
- Check the token counter below your prompt
- MPT supports up to 65k tokens
- GOAT supports up to 4k tokens
- Shorten your prompt if it exceeds the limit

## Performance Tips

### General Tips
1. **Choose the Right Model**:
   - Use MPT for very long stories and extended narratives
   - Use GOAT for high-quality, structured prose and character work
2. **Monitor Performance**: Enable "Show Memory Usage" to track resources
3. **Use Generation History**: Review previous outputs to refine prompts

### MPT-7B Optimization
- Excellent for continuing existing long stories
- Can handle detailed, lengthy prompts
- Faster generation than GOAT
- Works on CPU if needed (though slower)
- Temperature 0.7-0.9 recommended

### GOAT-70B Optimization
- Keep prompts focused and structured
- Use 300-1000 max tokens for best performance
- Excellent for novel chapters and character development
- Best with screenplay/script format instructions
- Temperature 0.8-1.0 for creative prose

## File Structure

```
kinder-kraft/
├── app.py              # Main Streamlit application with multi-model support
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore patterns
└── README.md          # This file
```

## Use Cases

### When to Use MPT-7B-StoryWriter
- Continuing long-running stories
- Generating extensive world-building content
- Creating detailed backstories
- Writing serialized fiction
- Any task requiring very long context

### When to Use GOAT-70B-Storytelling
- Writing novel chapters
- Developing complex characters
- Creating screenplay scenes
- Structured storytelling with specific arcs
- High-quality literary prose

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project uses models with their own license terms:
- [MPT-7B-StoryWriter License](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- [GOAT-70B-Storytelling License](https://huggingface.co/GOAT-AI/GOAT-70B-Storytelling)

## Acknowledgments

- [MosaicML](https://www.mosaicml.com/) for the MPT-7B-StoryWriter model
- [GOAT-AI](https://huggingface.co/GOAT-AI) for the GOAT-70B-Storytelling model
- [EleutherAI](https://www.eleuther.ai/) for the GPT-NeoX tokenizer
- [Streamlit](https://streamlit.io/) for the web framework
- [HuggingFace](https://huggingface.co/) for the Transformers library

## Support

For issues, questions, or suggestions, please open an issue in the repository.

## Changelog

### Version 2.0 - Multi-Model Support
- Added GOAT-70B-Storytelling model
- Model selection interface in sidebar
- Model-specific example prompts
- Dynamic parameter adjustment based on model
- Model comparison table
- Generation history tracking
- Enhanced performance metrics (tokens/second)
- GPU memory monitoring
- Improved error handling with model-specific messages

### Version 1.0 - Initial Release
- MPT-7B-StoryWriter support
- Basic story generation
- Parameter customization
- Token tracking
- Export functionality

---

**Happy Story Writing!** 📚✨
