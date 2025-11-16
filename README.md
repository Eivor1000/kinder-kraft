# AI Story Generator - MPT-7B-StoryWriter

A powerful Streamlit application for generating creative stories using the MPT-7B-StoryWriter-65k+ model from MosaicML.

## Features

- **Advanced Story Generation**: Leverages the MPT-7B-StoryWriter model capable of handling up to 65,000 tokens
- **Customizable Parameters**: Fine-tune generation with temperature, top-p, top-k, and max tokens controls
- **Example Prompts**: Quick-start with pre-made prompts across various genres
- **Token Tracking**: Monitor input and output token counts
- **Performance Metrics**: View generation time and optional memory usage
- **Export Stories**: Download generated stories as .txt files
- **GPU Support**: Automatic detection and use of CUDA-enabled GPUs for faster generation
- **User-Friendly Interface**: Clean, intuitive UI built with Streamlit

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended but not required)
- Minimum 16GB RAM (32GB recommended for better performance)
- ~15GB disk space for model download

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

**Note**: The first time you run the app, it will download the model (~15GB), which may take some time depending on your internet connection.

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the App

1. **Enter a Story Prompt**: Type your story idea or click on example prompts
2. **Adjust Parameters** (optional):
   - **Max Tokens**: Control the length of the generated story (100-2000)
   - **Temperature**: Adjust creativity (0.1 = focused, 2.0 = very creative)
   - **Top-p**: Control diversity via nucleus sampling (0.1-1.0)
   - **Top-k**: Limit vocabulary per step (0-100, 0 = disabled)
3. **Generate Story**: Click the "Generate Story" button
4. **Review & Download**: Read your generated story and download it if desired
5. **Clear**: Reset the interface to start fresh

### Example Prompts

The app includes built-in example prompts for:
- Fantasy Adventure
- Sci-Fi Mystery
- Historical Fiction
- Contemporary Drama
- Horror Thriller

## Configuration

### GPU vs CPU

The app automatically detects CUDA availability:
- **With GPU**: Uses bfloat16 precision for optimal performance
- **Without GPU**: Falls back to CPU with float32 precision (slower but functional)

### Memory Optimization

If you encounter out-of-memory errors:
- Reduce `max_tokens` parameter
- Close other memory-intensive applications
- Consider using a machine with more RAM
- Use GPU if available

## Technical Details

### Model Information

- **Model**: [mosaicml/mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- **Tokenizer**: [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)
- **Context Length**: Up to 65,000 tokens
- **Parameters**: 7 billion

### Key Features in Code

- **Model Caching**: `@st.cache_resource` decorator ensures model loads only once
- **Error Handling**: Comprehensive try-except blocks for robust operation
- **Session State**: Maintains state across interactions
- **Responsive UI**: Two-column layout for optimal user experience

## Troubleshooting

### Model fails to load

**Issue**: `RuntimeError: CUDA out of memory`
**Solution**:
- Ensure you have enough GPU memory (minimum 16GB recommended)
- Close other GPU-intensive applications
- Try running on CPU (automatic fallback)

### Slow generation on CPU

**Issue**: Story generation takes a long time
**Solution**:
- This is expected on CPU; consider using a GPU
- Reduce `max_tokens` parameter for faster results
- The model is 7B parameters and optimized for GPU use

### Import errors

**Issue**: `ModuleNotFoundError`
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Model download issues

**Issue**: Download interrupted or fails
**Solution**:
- Check internet connection
- Ensure sufficient disk space (~15GB)
- Try clearing HuggingFace cache: `~/.cache/huggingface/`

## Performance Tips

1. **Use GPU**: Significantly faster generation (10-50x speedup)
2. **Start Small**: Begin with lower max_tokens to test performance
3. **Optimize Parameters**:
   - Temperature 0.7-0.9 for balanced creativity
   - Top-p 0.9-0.95 for good quality
   - Top-k 40-60 for diverse vocabulary
4. **Batch Operations**: Generate multiple stories in one session to leverage cached model

## File Structure

```
kinder-kraft/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project uses the MPT-7B-StoryWriter model which is subject to its own license terms. Please review the [model card](https://huggingface.co/mosaicml/mpt-7b-storywriter) for details.

## Acknowledgments

- [MosaicML](https://www.mosaicml.com/) for the MPT-7B-StoryWriter model
- [EleutherAI](https://www.eleuther.ai/) for the GPT-NeoX tokenizer
- [Streamlit](https://streamlit.io/) for the web framework
- [HuggingFace](https://huggingface.co/) for the Transformers library

## Support

For issues, questions, or suggestions, please open an issue in the repository.

---

**Happy Story Writing!** 📚✨
