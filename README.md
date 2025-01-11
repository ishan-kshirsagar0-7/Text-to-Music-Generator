# Text-to-Music Generator

## Introduction

This application leverages the power of Facebook's Audiocraft library to create an interactive web application for audio generation and processing. Using cutting-edge deep learning models, this app provides an intuitive interface for users to generate music and sound effects based on textual descriptions.

## Outputs

Here are some examples :


https://github.com/ishan-kshirsagar0-7/Text-to-Music-Generator/assets/83628182/c751a8b1-8d9e-40a6-bcaa-b48632d513b6


https://github.com/ishan-kshirsagar0-7/Text-to-Music-Generator/assets/83628182/cfb7c0e9-b26e-47b2-a16e-425d815a2f82

## Demo (Try It Out)

1. Visit : [Click Here](https://huggingface.co/spaces/unpairedelectron07/Text-to-Music-Generator)
2. Adjust the slider first to the desired duration of the resulting audio, then click on the "Stop" button on the top-right corner.
3. Then enter your prompt, without worrying about the details, as the app will enhance it before passing it to the model.
4. Press "Ctrl + Enter" or "Cmd + Enter" to start the process. Note that it may take the app a maximum of 8 minutes to generate a 20 second audio.
5. Once the audio generates, you can then play it and download it.

## Methodology

The core of our application utilizes the Audiocraft library, specifically its MusicGen model, to convert text descriptions into high-quality audio. The Gemini-Pro model is used to enhance user's prompts to create a better description as the input for the MusicGen model. The app is built in Python, integrating technologies such as Streamlit for the web interface, PyTorch and torchaudio for model handling and audio processing, and Google's Generative AI for enhanced AI capabilities.

## Conclusion

This Text-to-Music Generator App showcases the potential of combining advanced audio generation models with user-friendly interfaces. It serves as a practical example of how deep learning can be applied to creative fields, making complex technology accessible to a broader audience.
