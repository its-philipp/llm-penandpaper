import streamlit as st
import re

from tts_utils import tts_manager
from image_gen_utils import generate_scene_image
import penandpaper as pnp

def extract_scene_description_from_tool_call(gm_text: str) -> str:
    """Extract scene description from generate_image tool call in GM message."""
    match = re.search(r'generate_image"?\s*[,\{][^\]]*scene_description"?\s*:\s*"([^"]+)"', gm_text)
    return match.group(1) if match else None

def clean_gm_message(gm_text: str) -> str:
    """Remove tool call JSON and meta commands from GM message."""
    # Remove tool call JSON (e.g., [{...}])
    cleaned = re.sub(r'\[\{.*?\}\]', '', gm_text, flags=re.DOTALL)
    # Remove explicit (Generate an image...) or similar meta commands
    cleaned = re.sub(r'\(Generate an image[^\)]*\)', '', cleaned)
    # Remove language/meta instruction echoes
    cleaned = re.sub(r'Language constraint:[^\n]*', '', cleaned)
    cleaned = re.sub(r'(Respond exclusively in English\.?|Antworten Sie ausschließlich auf Deutsch\.?)', '', cleaned)
    # Remove custom bracketed tags if leaked
    cleaned = re.sub(r'\[/?(LANGUAGE|INSTRUCTIONS)[^\]]*\]', '', cleaned)
    # Remove extra whitespace
    cleaned = re.sub(r'\n+', '\n', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()

def initialize_session_state():
    """Initialize Streamlit session state."""
    default_state = {
        'chat_history': [("GM", "Welcome to Eldoria Quest! Press 'Start Adventure' to begin.")],
        'auto_images': True,
        'last_image': None,
        'gm_initialized': False,
        'pending_gm_processing': False,
        'stop_voice_requested': False,
        'tts_image_ready': False,
        'pending_gm_text': None
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def stop_adventure():
    """Reset the adventure to initial state."""
    tts_manager.stop_tts()
    st.session_state.chat_history = [("GM", "Welcome to Eldoria Quest! Press 'Start Adventure' to begin.")]
    st.session_state.auto_images = True
    tts_manager.set_speaker("p230")
    st.session_state.last_image = None
    st.session_state.gm_initialized = False
    st.rerun()

def handle_special_commands(user_input: str) -> bool:
    """Handle special user commands. Returns True if command was handled."""
    user_input_lower = user_input.strip().lower()
    
    if user_input_lower == "stop voice":
        tts_manager.stop_tts()
        st.session_state.stop_voice_requested = True
        st.session_state.pending_gm_processing = False
        st.success("Voice stopped.")
        return True
        
    elif user_input_lower.startswith("voice "):
        try:
            speaker = user_input.strip().split(" ", 1)[1]
            tts_manager.set_speaker(speaker)
        except IndexError:
            st.warning("Please specify a speaker ID (e.g., voice p225).")
        return True
        
    elif user_input_lower == "toggle images":
        st.session_state.auto_images = not st.session_state.auto_images
        st.success(f"Automatic image generation: {'ON' if st.session_state.auto_images else 'OFF'}")
        return True
        
    elif user_input_lower == "generate image":
        last_gm_msg = next((msg for speaker, msg in reversed(st.session_state.chat_history) if speaker == "GM"), None)
        if last_gm_msg:
            st.info("Generating image of current scene...")
            try:
                scene_desc = extract_scene_description_from_tool_call(last_gm_msg)
                image_prompt = scene_desc if scene_desc else pnp.summarize_for_image(last_gm_msg)
                image_path = generate_scene_image(image_prompt)
                st.session_state.last_image = image_path
            except Exception as e:
                st.error(f"Image generation failed: {e}")
        else:
            st.warning("No GM narration found to generate an image.")
        return True
    
    return False

def process_gm_response(gm_text: str):
    """Process GM response with TTS and image generation."""
    # Guard: avoid re-processing the same GM text on reruns
    if st.session_state.get("tts_image_processed_for") == gm_text:
        return
    st.session_state["tts_image_processed_for"] = gm_text

    tts_manager.start_tts(gm_text)
    
    if st.session_state.auto_images:
        scene_desc = extract_scene_description_from_tool_call(gm_text)
        if scene_desc:
            image_prompt = scene_desc
        else:
            image_prompt = pnp.summarize_for_image(gm_text)
        image_path = generate_scene_image(image_prompt)
        st.session_state.last_image = image_path

    print("Processing GM:", gm_text[:20])

# Streamlit App Configuration
st.set_page_config(page_title="Eldoria Quest RPG", page_icon="🧙‍♂️", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #1a1a2e; color:#f5f6fa;}
    .stButton>button {background-color: #4e54c8; color: white; border-radius: 8px;}
    .stTextInput>div>div>input {background-color: #232946; color: #f5f6fa;}
    .stChatMessage {background-color: #232946; border-radius: 8px; margin-bottom: 8px;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

# Sidebar Controls
st.sidebar.title("🎛️ Controls")
# Language selection
lang_labels = {"English": "en", "Deutsch": "de"}
default_lang = st.session_state.get("language", "en")
selected_lang_label = st.sidebar.selectbox("Language", list(lang_labels.keys()), index=(0 if default_lang == "en" else 1))
selected_lang_code = lang_labels[selected_lang_label]
st.session_state["language"] = selected_lang_code
tts_manager.set_language(selected_lang_code)
# Model selection (Step 1: Ollama-only)
model_options = list(pnp.AVAILABLE_MODELS.keys())
active_model_key = pnp.get_current_model_key()
selected_model_key = st.sidebar.selectbox(
    "Select LLM Model",
    model_options,
    index=model_options.index(active_model_key) if active_model_key in model_options else 0,
)
if selected_model_key != active_model_key:
    pnp.set_model(selected_model_key)
    st.sidebar.success(f"Switched model to: {selected_model_key}")

voice_options = ["p230", "p225", "p226", "p227"]
selected_voice = st.sidebar.selectbox(
    "Select GM Voice", voice_options, index=voice_options.index(tts_manager.current_speaker)
)
tts_manager.set_speaker(selected_voice)

# Test TTS button
if st.sidebar.button("Test TTS"):
    sample = "Dies ist ein Test der Sprachausgabe." if st.session_state.get("language", "en") == "de" else "This is a test of the speech output."
    tts_manager.start_tts(sample)
    st.sidebar.info("Playing test TTS…")

if st.sidebar.button("Stop Voice"):
    tts_manager.stop_tts()
    st.session_state.stop_voice_requested = True
    st.session_state.pending_gm_processing = False
    st.session_state["tts_image_processed_for"] = None
    st.sidebar.success("Voice stopped.")

st.session_state.auto_images = st.sidebar.checkbox("Auto-generate images", value=st.session_state.auto_images)

# Main Title
st.title("🧙‍♂️ Eldoria Quest: The Shattered Moon of Valescourt")
st.markdown(
    "<div style='font-size:18px; color:#ffd700;'>An immersive AI-powered tabletop RPG adventure. Interact with your Game Master below!</div>",
    unsafe_allow_html=True
)

# Start Adventure Button
if not st.session_state.gm_initialized:
    if st.button("Start Adventure", use_container_width=True):
        with st.spinner("The Game Master is preparing your adventure..."):
            lang_code = st.session_state.get("language", "en")
            system_lang = ("Sprache: Antworte ausschließlich auf Deutsch." if lang_code == "de" else "Language: Respond exclusively in English.")
            system_content = f"{pnp.SYSTEM_MSG}\n\n{system_lang}"
            init_user = (
                ("Antworten Sie ausschließlich auf Deutsch." if lang_code == "de" else "Respond exclusively in English.") +
                "You are the Game Master. Set up an opening scene for a fantasy adventure. Describe where the player character finds themselves and what they can see around them. "
                "Use world_knowledge if you need information about the setting, and consider using generate_image for the opening scene, then ask the player what they want to do. "
            )
            try:
                startup = pnp.get_agent().invoke({
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": init_user}
                    ]
                }, config={"configurable": {"thread_id": "session1"}})
                gm_text = startup["messages"][-1].content
                st.session_state.chat_history.append(("GM", gm_text))
                st.session_state.gm_initialized = True
                st.session_state.pending_gm_text = gm_text
                st.rerun()
            except Exception as e:
                st.session_state.chat_history.append(("GM", "Welcome to the adventure! I'm ready to help with world knowledge, dice rolls, and visual scenes."))
                st.session_state.gm_initialized = True
                st.session_state.pending_gm_text = None
                st.rerun()
    
    # Show initial chat history
    for speaker, msg in st.session_state.chat_history:
        if speaker == "GM":
            st.markdown(f"<div class='stChatMessage'><b>GM:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='stChatMessage' style='background:#393e63;'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
    st.stop()

# Main Chat Display
for speaker, msg in st.session_state.chat_history:
    if speaker == "GM":
        display_msg = clean_gm_message(msg)
        st.markdown(f"<div class='stChatMessage'><b>GM:</b> {display_msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage' style='background:#393e63;'><b>You:</b> {msg}</div>", unsafe_allow_html=True)

# Display current image
if st.session_state.last_image:
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.image(st.session_state.last_image, caption="Current Scene", width=500)
    st.markdown("</div>", unsafe_allow_html=True)

# Manual image generation button (when auto-images is off)
if not st.session_state.auto_images:
    last_gm_msg = next((msg for speaker, msg in reversed(st.session_state.chat_history) if speaker == "GM"), None)
    if last_gm_msg and st.button("Generate Image for Current Scene", key="manual_image_btn"):
        st.info("Generating image of current scene...")
        try:
            scene_desc = extract_scene_description_from_tool_call(last_gm_msg)
            image_prompt = scene_desc if scene_desc else pnp.summarize_for_image(last_gm_msg)
            image_path = generate_scene_image(image_prompt)
            st.session_state.last_image = image_path
        except Exception as e:
            st.error(f"Image generation failed: {e}")

# User Input
user_input = st.text_input("Your action:", "", key="user_input")
submit = st.button("Send", use_container_width=True)

# Stop Adventure Button
if st.button("Stop Adventure", use_container_width=True):
    stop_adventure()

# Process user input
if submit and user_input.strip():
    st.session_state.chat_history.append(("You", user_input))
    
    # Handle special commands first
    if not handle_special_commands(user_input):
        # Process normal game input
        lang_code = st.session_state.get("language", "en")
        system_lang = ("Sprache: Antworte ausschließlich auf Deutsch." if lang_code == "de" else "Language: Respond exclusively in English.")
        system_content = f"{pnp.SYSTEM_MSG}\n\n{system_lang}"
        contextualized_input = (
            f"As the Game Master, respond to the player's action: '{user_input}'. "
            "Narrate what happens, use tools if needed for dice rolls, world information, or scene imagery, and ask what the player wants to do next. "
            + ("Antworten Sie ausschließlich auf Deutsch." if lang_code == "de" else "Respond exclusively in English.")
        )
        try:
            result = pnp.get_agent().invoke({
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": contextualized_input}
                ]
            }, config={"configurable": {"thread_id": "session1"}})
            gm_text = result["messages"][-1].content
            st.session_state.chat_history.append(("GM", gm_text))
            st.session_state.pending_gm_text = gm_text
            st.session_state.stop_voice_requested = False
            st.rerun()
        except Exception as e:
            st.session_state.chat_history.append(("GM", "I'm having trouble processing that request. Please try again."))
            st.session_state.pending_gm_text = None
            st.rerun()

# Process pending GM response (TTS and image generation)
if (st.session_state.pending_gm_text and not st.session_state.stop_voice_requested):
    process_gm_response(st.session_state.pending_gm_text)
    st.session_state.pending_gm_text = None 