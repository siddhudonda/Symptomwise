# app.py

import gradio as gr
from utils.health_analysis import get_youtube_transcript, analyze_transcript

def process_video_link(video_url):
    try:
        transcript = get_youtube_transcript(video_url)
        if not transcript:
            return "Transcript not found.", "", ""
        summary, results = analyze_transcript(transcript)
        return "Transcript fetched ✅", summary, f"Top Category: {results['labels'][0]} ({round(results['scores'][0]*100, 2)}%)"
    except Exception as e:
        return "Error fetching data", "", str(e)

iface = gr.Interface(
    fn=process_video_link,
    inputs=gr.Textbox(label="Enter YouTube Video URL"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Health Category")
    ],
    title="Symptomwise AI",
    description="Paste a YouTube link. Get transcript summary and health category."
)

if __name__ == "__main__":
    iface.launch()
