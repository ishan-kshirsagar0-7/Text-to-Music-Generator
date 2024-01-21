from audiocraft.models import MusicGen
import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import base64
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))
llm = genai.GenerativeModel("gemini-pro")

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

def generate_music_tensors(description, duration:int):
    print(f"Description: {description}")
    print(f"Duration: {duration}")
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    sample_rate = 32000
    save_path = "saved_audio/"

    assert samples.dim() == 2 or samples.dim() == 3
    samples = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def download_music(bin_file, file_label="File"):
    with open(bin_file, 'rb') as f:
        data = f.read()

    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon=":musical_note:",
    page_title="MusicGen"
)

def main():
    st.title("Text to Music Generation")

    with st.expander("View Details..."):

        st.write("This was built by https://github.com/ishan-kshirsagar0-7 using Meta's Audiocraft library. Enter the description of the music you want to generate, and set the duration with the slider given below. The longer the duration slider, the longer it will take to generate the music.")

    text_area = st.text_area("Enter your description...")
    time_slider = st.slider("Select time duration (in seconds)", 2, 20, 5)

    context = f"""Given the basic description of a prompt for a text-to-music generator below, enhance that prompt by using specific, direct, accurate and relevant vocabulary. This enhanced prompt must clearly assert and describe the kind of music user wants to generate, with the help of appropriate musical terminology or taxonomy. Craft a creative prompt that clearly explains the text-to-music model what music the user desires. DO NOT respond with anything other than the output prompt. You can be as creative as you like with the descriptions, but DO NOT make up details that the original prompt did not ask for. Also, make sure the description is not too lengthy, keep it concise. Your prompt must explain the flow of the music from start through the middle towards the finish, explicitly mentioning the way instruments are played and what they should sound like.

    ORIGINAL PROMPT : {text_area}
    YOUR OUTPUT PROMPT :
    """
    llm_result = llm.generate_content(context)
    prompt = llm_result.text

    if text_area and time_slider:
        st.json(
            {
                "Description": prompt,
                "Duration": time_slider
            }
        )

        st.subheader("Generated Music")

        music_tensors = generate_music_tensors(prompt, time_slider)
        print(f"Music Tensors: {music_tensors}")

        save_music_file = save_audio(music_tensors)

        audio_filepath = "saved_audio/audio_0.wav"
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes)
        st.markdown(download_music(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == '__main__':
    main()