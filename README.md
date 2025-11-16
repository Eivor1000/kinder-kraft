# AI Story Generator - Cloud API Edition

A lightweight Streamlit application for generating creative stories using HuggingFace Inference API. **No model downloads required** - all models run on HuggingFace's cloud servers!

## Features

### 🚀 No Downloads, No GPU Required!
- **Zero Setup**: No model downloads (saves ~15-140GB of disk space!)
- **No GPU Needed**: Models run on HuggingFace's powerful cloud infrastructure
- **Instant Start**: App launches in seconds, not minutes
- **Lower Costs**: No expensive GPU hardware required

### Multi-Model Support
- **MPT-7B-StoryWriter**: 65k token context window, perfect for ultra-long stories
- **GOAT-70B-Storytelling**: 70B parameters, specialized for novel writing and character development
- Easy model switching with one click
- Model comparison table to help choose the right model
- Model-specific example prompts and recommendations

### Core Features
- **HuggingFace API Integration**: Secure API token management
- **Customizable Parameters**: Temperature, top-p, top-k, and max tokens controls
- **Dynamic UI**: Interface adapts based on selected model
- **Token Tracking**: Estimated token counts with context usage indicators
- **Performance Metrics**: Generation time and tokens per second
- **Export Stories**: Download generated stories as .txt files with metadata
- **Generation History**: Track multiple generations in a session
- **User-Friendly Interface**: Clean, intuitive UI built with Streamlit

## Model Comparison

| Feature | MPT-7B-StoryWriter | GOAT-70B-Storytelling |
|---------|-------------------|----------------------|
| **Parameters** | 7 Billion | 70 Billion |
| **Context Window** | 65,536 tokens | 4,096 tokens |
| **Best For** | Long-form content, extended narratives | Structured plots, character development |
| **Speed** | Faster | Slower |
| **Hosting** | HuggingFace Cloud | HuggingFace Cloud |
| **Recommended Max Tokens** | 500-2000 | 300-1000 |
| **Specialty** | Ultra-long context | Novel-quality prose |

## Requirements

### System Requirements
- Python 3.8 or higher
- Internet connection
- **NO GPU required!**
- **NO model downloads!**
- Minimal disk space (< 100MB for app)

### HuggingFace Account
- Free HuggingFace account
- API token with read permissions

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

**Note**: Installation takes seconds! Only 2 lightweight dependencies needed.

### 4. Get your HuggingFace API Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "story-generator")
4. Select **"read"** permissions
5. Click "Generate token"
6. Copy the token (starts with `hf_...`)

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### First Time Setup

1. The app will prompt you to enter your HuggingFace API token
2. Paste your token in the sidebar (it will be stored in session state)
3. Token can be stored in environment variable `HF_TOKEN` or `HUGGINGFACE_TOKEN` for convenience

### Using the App

1. **Configure API Token**: Enter your HuggingFace token (first time only)
2. **Select a Model**: Choose between MPT-7B-StoryWriter or GOAT-70B-Storytelling
3. **Review Model Info**: Check the model description, context window, and recommendations
4. **Enter a Story Prompt**: Type your story idea or click on model-specific example prompts
5. **Adjust Parameters** (optional):
   - **Max Tokens**: Control the length (dynamically adjusted per model)
   - **Temperature**: Adjust creativity (0.1 = focused, 2.0 = very creative)
   - **Top-p**: Control diversity via nucleus sampling (0.1-1.0)
   - **Top-k**: Limit vocabulary per step (0-100, 0 = disabled)
6. **Generate Story**: Click the "Generate Story" button
7. **Review**: Check your generated story with performance metrics
8. **Download**: Export your story with metadata as a .txt file
9. **History**: View all stories generated in the current session

### Environment Variables (Optional)

Set your API token as an environment variable to avoid entering it each time:

```bash
# Linux/Mac
export HF_TOKEN="hf_your_token_here"
streamlit run app.py

# Windows
set HF_TOKEN=hf_your_token_here
streamlit run app.py
```

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

## API Information

### How It Works

- App makes HTTP requests to HuggingFace Inference API
- Models run on HuggingFace's cloud servers (not your machine)
- No model files downloaded or stored locally
- Responses streamed back to your app

### Rate Limits

- **Free Tier**: Limited requests per hour
- **Pro Tier**: Higher rate limits available
- If you hit rate limits, wait a few minutes or upgrade to Pro

### Costs

- **HuggingFace API**: Free tier available, Pro subscriptions for heavier usage
- **No local GPU costs**: Save on hardware and electricity!

## Technical Details

### Architecture

- **Frontend**: Streamlit web application
- **API Client**: `huggingface_hub.InferenceClient`
- **Authentication**: HuggingFace API token (read permissions)
- **Model Hosting**: HuggingFace Inference API (cloud-based)

### MPT-7B-StoryWriter

- **Model**: [mosaicml/mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- **Context Length**: Up to 65,536 tokens
- **Parameters**: 7 billion
- **Special Features**: Ultra-long context, continuation writing

### GOAT-70B-Storytelling

- **Model**: [GOAT-AI/GOAT-70B-Storytelling](https://huggingface.co/GOAT-AI/GOAT-70B-Storytelling)
- **Context Length**: 4,096 tokens
- **Parameters**: 70 billion
- **Special Features**: Novel-quality prose, character development

### Key Features in Code

- **API Token Management**: Secure storage in session state or environment variables
- **Inference Client**: Uses `huggingface_hub.InferenceClient` for API calls
- **Error Handling**: Specific handling for authorization, rate limits, timeouts
- **Token Estimation**: Approximate token counting (4 chars per token)
- **Session State**: Maintains prompts, stories, and generation history
- **Model-Specific Prompts**: Tailored examples for each model

## Troubleshooting

### API Token Issues

**Issue**: "Authorization error"
**Solution**:
- Check token is valid at https://huggingface.co/settings/tokens
- Ensure token has "read" permissions
- Token should start with `hf_`
- Try generating a new token

### Rate Limit Errors

**Issue**: "Rate limit exceeded"
**Solution**:
- Wait 5-10 minutes before trying again
- Consider upgrading to HuggingFace Pro for higher limits
- Reduce max_tokens to make fewer API calls

### Model Loading

**Issue**: "Model is currently loading"
**Solution**:
- Wait 30-60 seconds and try again
- This happens when the model needs to "warm up" on HuggingFace servers
- Subsequent requests will be faster

### Timeout Errors

**Issue**: "Request timeout"
**Solution**:
- Reduce max_tokens parameter
- Check your internet connection
- Try again in a few moments

### Connection Issues

**Issue**: Cannot connect to HuggingFace API
**Solution**:
- Check your internet connection
- Verify HuggingFace is not down: https://status.huggingface.co
- Check firewall settings

## Performance Tips

### General Tips
1. **Choose the Right Model**:
   - Use MPT for very long stories and extended narratives
   - Use GOAT for high-quality, structured prose and character work
2. **Start with Lower Max Tokens**: Test with 300-500 tokens before going higher
3. **Monitor History**: Review previous generations to refine prompts

### MPT-7B Optimization
- Excellent for continuing existing long stories
- Can handle detailed, lengthy prompts
- Faster generation than GOAT
- Temperature 0.7-0.9 recommended

### GOAT-70B Optimization
- Keep prompts focused and structured
- Use 300-1000 max tokens for best performance
- Excellent for novel chapters and character development
- Best with screenplay/script format instructions
- Temperature 0.8-1.0 for creative prose

## Advantages of API Approach

### vs. Local Model Loading

| Aspect | API Approach (This App) | Local Loading |
|--------|------------------------|---------------|
| **Disk Space** | < 100MB | 15-140GB |
| **RAM Required** | < 1GB | 16-80GB |
| **GPU Required** | No | Yes (for good speed) |
| **Setup Time** | < 1 minute | 10-60 minutes |
| **Initial Load** | Instant | 1-5 minutes |
| **Maintenance** | Zero | Updates, drivers, etc. |
| **Portability** | Any device | Powerful workstation only |
| **Cost** | API fees (low for casual use) | Expensive GPU hardware |

## File Structure

```
kinder-kraft/
├── app.py              # Main Streamlit application (API-based)
├── requirements.txt    # Minimal dependencies (2 packages)
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

## Security & Privacy

- API tokens stored only in session state (not persisted)
- Can use environment variables for token management
- Prompts and stories sent to HuggingFace servers
- HuggingFace Privacy Policy: https://huggingface.co/privacy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project uses models with their own license terms:
- [MPT-7B-StoryWriter License](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- [GOAT-70B-Storytelling License](https://huggingface.co/GOAT-AI/GOAT-70B-Storytelling)

## Acknowledgments

- [MosaicML](https://www.mosaicml.com/) for the MPT-7B-StoryWriter model
- [GOAT-AI](https://huggingface.co/GOAT-AI) for the GOAT-70B-Storytelling model
- [HuggingFace](https://huggingface.co/) for the Inference API and infrastructure
- [Streamlit](https://streamlit.io/) for the web framework

## Support

For issues, questions, or suggestions, please open an issue in the repository.

## Changelog

### Version 3.0 - Cloud API Edition
- **BREAKING CHANGE**: Switched from local model loading to HuggingFace Inference API
- No model downloads required
- No GPU required
- Instant startup (< 1 second)
- API token management with environment variable support
- Token estimation instead of exact counting
- Enhanced error handling for API-specific issues (rate limits, authorization, etc.)
- Minimal dependencies (only streamlit and huggingface-hub)
- Significant reduction in disk space and memory requirements

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
- MPT-7B-StoryWriter support (local loading)
- Basic story generation
- Parameter customization
- Token tracking
- Export functionality

---

**Happy Story Writing!** 📚✨

**No downloads. No GPU. Just great stories.** 🚀
