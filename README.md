# 🧙‍♂️ Eldoria Quest: The Shattered Moon of Valescourt

An immersive AI-powered tabletop RPG adventure that combines the magic of storytelling with cutting-edge AI technology. Experience a dynamic, interactive fantasy world where every decision shapes your journey.

## 🌟 Features

### 🎭 **Dynamic AI Game Master**
- **Intelligent Storytelling**: Powered by Mistral 7B Instruct model for natural, engaging narrative
- **Adaptive Responses**: The AI GM remembers your choices and adapts the story accordingly
- **Rich Character Interactions**: Meet and interact with memorable NPCs like King Alaric Artingale, Queen Aelinthra Moonshade, and Captain Selina Marrash

### 🎲 **Authentic RPG Mechanics**
- **Dice Rolling System**: Automatic dice rolls for skill checks (Wissen, Handeln, Soziales)
- **World Knowledge Integration**: AI searches through game lore to provide accurate information
- **Progressive Storytelling**: Three parallel story threads leading to the ultimate Lunar Convergence

### 🎨 **Immersive Visual Experience**
- **AI-Generated Art**: Automatic scene generation using Stable Diffusion
- **Dynamic Image Creation**: Visual representations of key moments and locations
- **Atmospheric Imagery**: Fantasy artwork that brings the world of Eldoria to life
- **Multi-GPU Support**: Optimized for NVIDIA CUDA, AMD ROCm, and Apple Metal

### 🔊 **Audio Enhancement**
- **Text-to-Speech**: Multiple voice options for the Game Master
- **Edge TTS Integration**: High-quality, natural-sounding speech synthesis
- **Voice Control**: Start, stop, and switch between different GM voices

### 🎮 **Interactive Web Interface**
- **Streamlit Web App**: Beautiful, responsive interface
- **Real-time Chat**: Seamless conversation with your AI Game Master
- **Session Management**: Save and resume your adventure progress

### ⚡ **Performance Optimizations**
- **AMD GPU Support**: Full compatibility with AMD ROCm for accelerated image generation
- **Smart Device Detection**: Automatic selection of the best available GPU (NVIDIA, AMD, Apple Metal, CPU)
- **Memory Management**: Efficient resource usage and cache clearing
- **Optimized Code**: Clean, efficient codebase with type hints and improved error handling

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** installed and running locally
- **Mistral 7B Instruct** model downloaded
- **GPU Support** (optional but recommended):
  - **NVIDIA**: CUDA-compatible GPU
  - **AMD**: ROCm-compatible GPU (RX 5000+ series)
  - **Apple**: M1/M2 Mac with Metal support

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-penandpaper
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

3. **Start Ollama and download the model**
   ```bash
   ollama serve
   ollama pull mistral:7b-instruct
   ```

4. **Launch the application**
   ```bash
   uv run streamlit run penandpaper_streamlit.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

#### LiteLLM (optional router)

Start a LiteLLM server to unify local Ollama models and OpenAI:

1) Create `litellm.yaml`:

```
model_list:
  - model_name: local-mistral
    litellm_params:
      model: ollama/mistral:7b-instruct
      api_base: http://localhost:11434
  - model_name: local-llama3
    litellm_params:
      model: ollama/llama3:8b
      api_base: http://localhost:11434
  - model_name: openai-gpt4o-mini
    litellm_params:
      model: openai/gpt-4o-mini

litellm_settings:
  callbacks: ["langfuse"]
```

2) Run LiteLLM:

```
uv run litellm --config litellm.yaml --host 0.0.0.0 --port 4000
```

3) In a separate shell, export for the app:

```
export LITELLM_BASE_URL=http://localhost:4000
export LITELLM_API_KEY=litellm-proxy-key
```

4) Use router models from the sidebar (`router/local-mistral`, `router/local-llama3`, `router/openai-gpt4o-mini`).

## 🎯 How to Play

### Starting Your Adventure
1. Click **"Start Adventure"** to begin your journey
2. Choose your preferred **GM voice** from the sidebar
3. Enable **auto-image generation** for visual scenes
4. Begin your conversation with the AI Game Master

### Gameplay Mechanics
- **Respond naturally**: Type your actions and dialogue as you would in a real RPG
- **Ask questions**: Inquire about the world, characters, or your current situation
- **Make decisions**: Your choices influence the story's direction
- **Explore freely**: The AI adapts to your playstyle and preferences

### Available Commands
- **Voice Control**: Use the sidebar to stop speech or change GM voice
- **Image Generation**: Toggle automatic scene visualization
- **Adventure Reset**: Start fresh with the "Stop Adventure" button

## 🏗️ Technical Architecture

### Core Components

#### **AI Integration**
- **LangChain**: Framework for LLM orchestration and tool integration
- **LangGraph**: Agent-based architecture for complex reasoning
- **LlamaIndex**: RAG (Retrieval Augmented Generation) for world knowledge
- **Ollama**: Local LLM inference with Mistral 7B Instruct

#### **Visual Generation**
- **Stable Diffusion**: AI image generation for scene visualization
- **Multi-GPU Support**: NVIDIA CUDA, AMD ROCm, Apple Metal, and CPU fallback
- **Automatic Prompting**: Intelligent scene description extraction and enhancement
- **Memory Optimization**: Efficient cache management and resource cleanup

#### **Audio System**
- **Edge TTS**: Microsoft's text-to-speech service
- **Multi-Voice Support**: Four distinct GM voice options
- **Threading**: Non-blocking audio playback with improved resource management

#### **Web Interface**
- **Streamlit**: Modern web application framework
- **Session Management**: Persistent state across interactions
- **Responsive Design**: Optimized for various screen sizes

### Project Structure
```
llm-penandpaper/
├── penandpaper.py              # Core game logic and AI integration
├── penandpaper_streamlit.py    # Web interface and user interaction
├── tts_utils.py               # Text-to-speech functionality
├── image_gen_utils.py         # AI image generation with multi-GPU support
├── game_docs/                 # World lore and game materials
│   ├── world_description.md   # Detailed world information
│   ├── storyline.txt         # Main campaign narrative
│   └── Regelwerk.pdf         # Game rules and mechanics
├── generated_images/          # AI-generated scene images
├── pyproject.toml            # Project dependencies with AMD GPU support
└── uv.lock                   # Locked dependency versions
```

## 🎮 Game World: Eldoria

### Setting
**Valescourt** - A mystical realm where the ancient Crystalline Moondials have been shattered, threatening the balance of magic and reality. Your adventure begins at the **Grand Midsummer Festival** in the bustling port city of **Valecross**.

### Story Threads
1. **Ironwood Forest** - The Dial of Roots
2. **Shattered Isles** - The Dial of Tides  
3. **Ember Wastes** - The Dial of Flames

### Key Characters
- **King Alaric Artingale**: Wise ruler of Valescourt
- **Queen Aelinthra Moonshade**: Mysterious and powerful
- **Captain Selina Marrash**: Your guide and ally

## 🔧 Configuration

### Environment Variables
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `AUTO_GENERATE_IMAGES`: Enable/disable automatic image generation
- `IMAGE_KEYWORDS`: Keywords that trigger image generation

### Model Settings
- **LLM**: Mistral 7B Instruct (optimized for instruction following)
- **Embeddings**: BAAI/bge-small-en-v1.5 (efficient semantic search)
- **Image Generation**: Stable Diffusion v1.5 (high-quality fantasy art)

### GPU Support
- **NVIDIA CUDA**: Automatic detection and optimization
- **AMD ROCm**: Full support for AMD GPUs via ROCm
- **Apple Metal**: Optimized for M1/M2 Macs
- **CPU Fallback**: Reliable performance on any system

## 🛠️ Development

### Adding New Features
1. **Tools**: Add new `@tool` decorated functions in `penandpaper.py`
2. **UI Elements**: Extend the Streamlit interface in `penandpaper_streamlit.py`
3. **World Content**: Add documents to `game_docs/` for RAG integration

### Dependencies
This project uses **uv** for fast, reliable dependency management:
- **Python 3.11+** required
- **All dependencies** are pinned in `pyproject.toml`
- **Reproducible builds** via `uv.lock`
- **AMD GPU support** via optional dependencies

### Performance Optimization
- **Multi-GPU Support**: Automatic device detection and optimization
- **Memory Management**: Efficient cache clearing and resource cleanup
- **Async Operations**: Non-blocking TTS and image generation
- **Code Optimization**: Type hints, improved error handling, and clean architecture

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the powerful language model
- **Stability AI** for Stable Diffusion
- **Microsoft** for Edge TTS
- **LangChain** and **LangGraph** communities
- **Streamlit** for the web framework
- **AMD** for ROCm GPU support
- **PyTorch** for multi-GPU compatibility

---

**Ready to begin your adventure?** Start the application and let the magic of AI-powered storytelling transport you to the world of Eldoria! 🧙‍♂️✨

## 📸 Screenshots
![screenshot1.png](/screenshots/screenshot1.png)
![screenshot2.png](/screenshots/screenshot2.png)
![screenshot3.png](/screenshots/screenshot3.png)
![screenshot4.png](/screenshots/screenshot4.png)
![screenshot5.png](/screenshots/screenshot5.png)
