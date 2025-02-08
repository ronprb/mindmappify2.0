import yt_dlp
import os
from pydub import AudioSegment
import pygraphviz as pgv
from transformers import pipeline
import whisper
import streamlit as st
from PIL import Image
import time
import subprocess
from openai import OpenAI

# Function to clear previous files
def clear_previous_files():
    for file in os.listdir("."):
        if file.startswith("audio") or file in ["mind_map.md", "mind_map.html"]:
            os.remove(file)

# Function to download audio from YouTube
def download_audio_yt_dlp(video_url):
    clear_previous_files()  # Clear existing files
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',  # Output filename format
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Function to convert audio to WAV using pydub
def convert_audio_to_wav(input_file, output_file="audio.wav"):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# Function to summarize text
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    max_chunk_size = 1024
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    summarized_text = " ".join(summaries)
    return summarized_text


# Function to generate mind map markdown using Claude
def generate_mind_map_md(summarized_text):
    api_key = os.getenv("OPENAI_API_KEY")  # Load from environment variable
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Given the following summary, generate a mind map in markdown format, bold the important things:
    
    {summarized_text}
    
    Output the mind map with a clear hierarchy, using bullet points for branches.
    """

    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo" for a faster, cheaper option
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.7
    )

    mind_map = response.choices[0].message.content.strip()

    with open("mind_map.md", "w") as f:
        f.write(mind_map)

    return "mind_map.md"

# Function to convert markdown to HTML using markmap-cli
def convert_md_to_html():
    subprocess.run(["markmap", "mind_map.md", "-o", "mind_map.html"])

# Streamlit Interface
def main():
    st.title("MindMapify for YouTube \n(CS688 Project - Rony Purba, 2024)")
    
    # Video URL input
    video_url = st.text_input("Enter YouTube Video URL")
    
    if video_url:
        st.write("Downloading audio...")
        
        # Show progress bar while downloading
        with st.spinner('Downloading audio...'):
            download_audio_yt_dlp(video_url)
        
        downloaded_files = os.listdir(".")
        audio_file = None
        for file in downloaded_files:
            if file.endswith(".mp3") or file.endswith(".m4a") or file.endswith(".webm"):
                audio_file = file
                break
        
        if audio_file:
            # Audio processing and transcription
            st.write(f"Converting and transcribing {audio_file}...")
            audio_wav = convert_audio_to_wav(audio_file)
            text = transcribe_audio(audio_wav)
            st.write("Transcription completed.")
            
            # Show summary generation progress
            st.write("Generating summary...")
            summary = summarize_text(text)
            st.write(f'*{summary}*')
            
            # Mind map generation
            st.write("Generating mind map...")
            mind_map_md_path = generate_mind_map_md(summary)
            convert_md_to_html()
            st.write("Mind map generated.")

            # Display mind map
            with open("mind_map.html", "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
            
            # Download buttons in columns for better layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="Download Summary as TXT",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
            
            with col2:
                st.download_button(
                    label="Download Mind Map as HTML",
                    data=open("mind_map.html", "rb").read(),
                    file_name="mind_map.html",
                    mime="application/html"
                )
            
            with col3:
                st.download_button(
                    label="Download Audio File",
                    data=open(audio_file, "rb").read(),
                    file_name=audio_file,
                    mime="audio/mpeg"
                )
        else:
            st.error("No audio file found. Please check the download process.")

if __name__ == "__main__":
    main()