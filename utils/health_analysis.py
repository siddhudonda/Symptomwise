# utils/health_analysis.py

import os
import googleapiclient.discovery
from transformers import pipeline

# Initialize pipelines globally to avoid repeated loading
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_youtube_transcript(video_url):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.getenv("YOUTUBE_API_KEY")  # Make sure this is set in your environment

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    video_id = video_url.split("v=")[-1]
    request = youtube.captions().list(part="snippet", videoId=video_id)
    response = request.execute()

    transcript = ""
    if "items" in response:
        for item in response["items"]:
            transcript += item["snippet"].get("text", "")
    return transcript

def analyze_transcript(transcript):
    summary = summarizer(transcript[:1000])[0]["summary_text"]
    health_tags = [
        "Mental Health", "Physical Fitness", "Nutrition",
        "Sleep", "Work-life balance", "Emotional well-being"
    ]
    results = classifier(summary, candidate_labels=health_tags)
    return summary, results
