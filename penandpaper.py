import os
import random
import re

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from tts_utils import tts_manager
from image_gen_utils import generate_scene_image

# System message for the AI Game Master
SYSTEM_MSG = """You are an experienced Dungeon Master (Game Master) running the "Eldoria Quest: The Shattered Moon of Valescourt" tabletop RPG campaign. Your role is to:

1. Narrate the story and describe scenes vividly, following the storyline from the game_docs
2. Guide the player through the adventure starting at the Grand Midsummer Festival in Valecross
3. Control NPCs like King Alaric Artingale, Queen Aelinthra Moonshade, and Captain Selina Marrash
4. Present the three parallel story threads: Ironwood Forest (Dial of Roots), Shattered Isles (Dial of Tides), and Ember Wastes (Dial of Flames)
5. Use the world_knowledge tool to get information about the game world and storyline when needed
6. Use the roll_dice tool when game mechanics require dice rolls (Wissen, Handeln, Soziales probes)
7. Use the generate_image tool when describing new locations, important scenes, or significant visual moments
8. Track the player's progress toward reassembling the three Crystalline Moondials
9. Always end your responses by asking the player what they want to do next

You should speak in second person ("You see...", "You notice...") and maintain an immersive, storytelling tone. Keep responses concise but engaging. Follow the storyline structure and guide the player toward the final Lunar Convergence at the Moonglints Obelisk.

When describing visual scenes, be detailed about the environment, lighting, characters, and atmosphere to help create vivid imagery."""

# Configuration constants
AUTO_GENERATE_IMAGES = True
IMAGE_KEYWORDS = ["see", "notice", "view", "appears", "landscape", "building", "creature", "character", "room", "forest", "castle"]

# Available local models (Step 1: Ollama only)
AVAILABLE_MODELS = {
    "ollama/mistral": {
        "provider": "ollama",
        "model": "mistral:7b-instruct",
    },
    "ollama/llama3": {
        "provider": "ollama",
        "model": "llama3:8b",
    },
    # Router (LiteLLM) aliases
    "router/local-mistral": {
        "provider": "litellm",
        "model": "local-mistral",
    },
    "router/local-llama3": {
        "provider": "litellm",
        "model": "local-llama3",
    },
    "router/openai-gpt4o-mini": {
        "provider": "litellm",
        "model": "openai-gpt4o-mini",
    },
}

_current_model_key = "ollama/mistral"

# Initialize components
def _create_llm(model_key: str):
    """Create an LC LLM instance based on a model key."""
    model_info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["ollama/mistral"])
    provider = model_info["provider"]
    if provider == "ollama":
        return ChatOllama(model=model_info["model"], base_url="http://localhost:11434")
    if provider == "litellm":
        # Route via LiteLLM proxy compatible with OpenAI API
        base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
        # Ensure OpenAI-compatible path
        if not base_url.rstrip('/').endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        api_key = os.environ.get("LITELLM_API_KEY", "litellm-proxy-key")
        return ChatOpenAI(model=model_info["model"], base_url=base_url, api_key=api_key)
    raise ValueError(f"Unsupported provider for model_key={model_key}")

def initialize_components(model_key: str = _current_model_key):
    """Initialize LLM, embeddings, and knowledge base."""
    # Initialize LangChain LLM
    llm = _create_llm(model_key)
    
    # Configure LlamaIndex
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Load world docs & build vector store index
    script_dir = os.path.dirname(os.path.abspath(__file__))
    game_docs_path = os.path.join(script_dir, "game_docs")
    documents = SimpleDirectoryReader(input_dir=game_docs_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    return llm, query_engine

# Initialize components
llm, query_engine = initialize_components(_current_model_key)

# Tool functions
@tool
def roll_dice(command: str) -> str:
    """Roll dice in format d6, D20, etc."""
    match = re.fullmatch(r"[dD](\d+)", command.strip())
    if not match:
        return "Invalid format. Usage: d6, D20, etc."
    return str(random.randint(1, int(match.group(1))))

@tool
def world_knowledge(query: str) -> str:
    """Search world lore for relevant info."""
    return str(query_engine.query(query))

@tool
def generate_image(scene_description: str) -> str:
    """Generate an image for the current scene description."""
    try:
        image_path = generate_scene_image(scene_description)
        return f"Image generated and saved to: {image_path}"
    except Exception as e:
        return f"Failed to generate image: {str(e)}"

def _create_agent(model_key: str):
    """Create a LangGraph REACT agent for the given model key."""
    local_llm = _create_llm(model_key)
    Settings.llm = local_llm  # ensure LlamaIndex uses the same LLM
    # Workaround: some llama3 builds return 400 with certain payload patterns.
    # For llama3 (direct or via router), construct agent without bound tools.
    is_llama3 = model_key in ("ollama/llama3", "router/local-llama3")
    # Disable tools for router/mistral to prevent tool-calling loops
    is_router_mistral = model_key == "router/local-mistral"
    tool_list = [] if is_llama3 or is_router_mistral else [roll_dice, world_knowledge, generate_image]
    return create_react_agent(
        model=local_llm,
        tools=tool_list,
        checkpointer=memory
    )

# Initialize LangGraph REACT agent
memory = InMemorySaver()
agent = _create_agent(_current_model_key)

def set_model(model_key: str) -> str:
    """Rebuild the global LLM and agent with the selected model key. Returns the active key."""
    global llm, agent, _current_model_key
    _current_model_key = model_key if model_key in AVAILABLE_MODELS else "ollama/mistral"
    llm = _create_llm(_current_model_key)
    Settings.llm = llm
    # Respect the same llama3 workaround when rebuilding
    is_llama3 = _current_model_key in ("ollama/llama3", "router/local-llama3")
    is_router_mistral = _current_model_key == "router/local-mistral"

    tool_list = [] if is_llama3 or is_router_mistral else [roll_dice, world_knowledge, generate_image]
    agent = create_react_agent(
        model=llm,
        tools=tool_list,
        checkpointer=memory
    )
    return _current_model_key

def get_agent():
    """Return the current global agent instance."""
    return agent

def get_current_model_key() -> str:
    """Return the currently active model key."""
    return _current_model_key

def should_generate_image(text: str) -> bool:
    """Determine if the GM's response warrants an image."""
    if not AUTO_GENERATE_IMAGES:
        return False
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in IMAGE_KEYWORDS)

def process_gm_response(gm_text: str) -> None:
    """Process GM response and potentially generate an image."""
    print("GM>", gm_text)
    
    if should_generate_image(gm_text):
        print("\n[Generating scene image...]")
        try:
            image_path = generate_scene_image(gm_text)
            print(f"[Image saved: {image_path}]")
        except Exception as e:
            print(f"[Image generation failed: {e}]")
    
    tts_manager.start_tts(gm_text)

def print_controls() -> None:
    """Print available controls."""
    print("\n=== Controls ===")
    print("• Type 'stop voice' to stop current speech")
    print("• Type 'voice [speaker]' to change voice")
    print("• Type 'toggle images' to enable/disable automatic image generation")
    print("• Type 'generate image' to manually create an image of the current scene")
    print("• Available voices: p230=Alex, p225=Daniel, p226=Tom, p227=Ralph")
    print("================\n")

def summarize_for_image(narration: str) -> str:
    """Summarize narration for image generation using the underlying chat LLM only (no tools)."""
    prompt = (
        "You are assisting with image prompt preparation. "
        "Return ONLY a succinct visual summary (1-2 sentences) of the scene, focusing on setting, characters, objects, and atmosphere. "
        "Do NOT include meta instructions, rationale, or extra commentary.\n\n"
        f"Scene: {narration}\n\nSummary:"
    )
    # Call the chat model directly to avoid agent tool calls causing image-gen loops
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))
    return content.strip()

def main():
    """Main game loop."""
    print("===== Welcome to your RPG! =====")
    try:
        init_msg = f"{SYSTEM_MSG}\n\nYou are the Game Master. Set up an opening scene for a fantasy adventure. Describe where the player character finds themselves and what they can see around them. Use world_knowledge if you need information about the setting, and consider using generate_image for the opening scene, then ask the player what they want to do."
        startup = agent.invoke(
            {"messages": [{"role": "user", "content": init_msg}]},
            config={"configurable": {"thread_id": "session1"}}
        )
        gm_text = startup["messages"][-1].content
        process_gm_response(gm_text)
    except Exception as e:
        print(f"Error during startup: {e}")
        print("GM> Welcome to the adventure! I'm ready to help with world knowledge, dice rolls, and visual scenes.")

    print_controls()

    while True:
        user_input = input("You> ").strip().lower()
        
        if user_input in ("exit", "quit"):
            tts_manager.stop_tts()
            print("Session ended.")
            break
            
        if user_input == "stop voice":
            tts_manager.stop_tts()
            print("Voice stopped.")
            continue
            
        if user_input.startswith("voice "):
            try:
                speaker = user_input.split(" ", 1)[1]
                tts_manager.set_speaker(speaker)
            except IndexError:
                print("Please specify a speaker ID (e.g., voice p225).")
            continue
            
        if user_input == "toggle images":
            global AUTO_GENERATE_IMAGES
            AUTO_GENERATE_IMAGES = not AUTO_GENERATE_IMAGES
            print(f"Automatic image generation: {'ON' if AUTO_GENERATE_IMAGES else 'OFF'}")
            continue
            
        if user_input == "generate image":
            print("[Generating image of current scene...]")
            try:
                image_path = generate_scene_image("fantasy RPG scene, medieval fantasy setting, detailed environment")
                print(f"[Image saved: {image_path}]")
            except Exception as e:
                print(f"[Image generation failed: {e}]")
            continue
            
        try:
            contextualized_input = f"As the Game Master, respond to the player's action: '{user_input}'. Narrate what happens, use tools if needed for dice rolls, world information, or scene imagery, and ask what the player wants to do next."
            result = agent.invoke(
                {"messages": [{"role": "user", "content": contextualized_input}]},
                config={"configurable": {"thread_id": "session1"}}
            )
            gm_text = result["messages"][-1].content
            process_gm_response(gm_text)
        except Exception as e:
            print(f"Error: {e}")
            print("GM> I'm having trouble processing that request. Please try again.")

if __name__ == "__main__":
    main()