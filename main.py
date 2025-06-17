import gradio as gr
from transformers import pipeline
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# Cache the Hugging Face models to avoid reloading
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extended list of categories (40+ symptoms and diseases)
categories = [
    "headache", "migraine", "stomach pain", "dizziness and nausea",
    "chest pain", "shortness of breath", "fatigue", "fever",
    "cough", "sore throat", "runny nose", "muscle pain",
    "joint pain", "back pain", "abdominal pain", "diarrhea",
    "constipation", "vomiting", "heartburn", "indigestion",
    "blurred vision", "eye pain", "ear pain", "hearing loss",
    "tinnitus", "sinus pain", "nasal congestion", "toothache",
    "jaw pain", "memory loss", "confusion", "anxiety",
    "depression", "insomnia", "tremors", "seizures",
    "numbness", "tingling", "rash", "itching",
    "swelling", "high blood pressure", "low blood pressure",
    "irregular heartbeat", "palpitations", "allergies",
    "asthma", "pneumonia", "bronchitis", "urinary tract infection",
    "kidney stones", "menstrual cramps", "menopause symptoms",
    "erectile dysfunction", "prostate problems", "osteoporosis",
    "arthritis", "diabetes", "thyroid problems", "anemia",
    "cancer", "stroke", "heart attack", "Alzheimer's disease",
    "Parkinson's disease", "epilepsy", "multiple sclerosis",
    "lupus", "fibromyalgia", "chronic fatigue syndrome",
    "irritable bowel syndrome", "Crohn's disease", "ulcerative colitis",
    "gout", "psoriasis", "eczema", "acne", "rosacea",
    "shingles", "herpes", "HIV/AIDS", "hepatitis",
    "cirrhosis", "gallstones", "pancreatitis", "appendicitis",
    "hernia", "varicose veins", "deep vein thrombosis",
    "lymphoma", "leukemia", "osteosarcoma", "melanoma",
    "cataracts", "glaucoma", "macular degeneration",
    "vertigo", "motion sickness", "food poisoning",
    "dehydration", "heat stroke", "hypothermia",
    "frostbite", "sunburn", "poison ivy", "insect bites",
    "snake bites", "spider bites", "rabies", "tetanus",
    "malaria", "dengue fever", "tuberculosis", "COVID-19",
    "influenza", "common cold", "pneumonia", "bronchiolitis",
    "whooping cough", "measles", "mumps", "rubella",
    "chickenpox", "smallpox", "polio", "meningitis",
    "encephalitis", "Lyme disease", "Rocky Mountain spotted fever",
    "West Nile virus", "Zika virus", "Ebola virus",
    "yellow fever", "cholera", "typhoid fever", "diphtheria",
    "pertussis", "scarlet fever", "rheumatic fever",
    "endocarditis", "myocarditis", "pericarditis",
    "cardiomyopathy", "heart failure", "angina",
    "atherosclerosis", "peripheral artery disease",
    "aneurysm", "varicose veins", "hemorrhoids",
    "diverticulitis", "celiac disease", "lactose intolerance",
    "peptic ulcer", "gastroesophageal reflux disease",
    "hiatal hernia", "gallbladder disease", "pancreatic cancer",
    "liver cancer", "kidney cancer", "bladder cancer",
    "prostate cancer", "testicular cancer", "ovarian cancer",
    "cervical cancer", "endometrial cancer", "breast cancer",
    "lung cancer", "skin cancer", "brain cancer",
    "bone cancer", "soft tissue sarcoma", "lymphoma",
    "leukemia", "myeloma", "neuroendocrine tumors",
    "carcinoid tumors", "mesothelioma", "Kaposi's sarcoma",
    "Wilms tumor", "retinoblastoma", "neuroblastoma",
    "Ewing sarcoma", "osteosarcoma", "chondrosarcoma",
    "rhabdomyosarcoma", "liposarcoma", "leiomyosarcoma",
    "synovial sarcoma", "angiosarcoma", "fibrosarcoma",
    "malignant peripheral nerve sheath tumor", "gastrointestinal stromal tumor",
    "desmoid tumor", "pheochromocytoma", "paraganglioma",
    "adrenal cancer", "pituitary tumor", "thyroid cancer",
    "parathyroid cancer", "thymoma", "thymic carcinoma",
    "carcinoid tumor", "islet cell tumor", "pancreatic neuroendocrine tumor",
    "merkel cell carcinoma", "basal cell carcinoma",
    "squamous cell carcinoma", "melanoma", "actinic keratosis",
    "seborrheic keratosis", "dermatofibroma", "lipoma",
    "hemangioma", "lymphangioma", "neurofibroma",
    "schwannoma", "meningioma", "glioma", "astrocytoma",
    "oligodendroglioma", "ependymoma", "medulloblastoma",
    "pineal tumor", "craniopharyngioma", "pituitary adenoma",
    "acromegaly", "Cushing's syndrome", "Addison's disease",
    "hyperthyroidism", "hypothyroidism", "goiter",
    "thyroiditis", "Hashimoto's thyroiditis", "Graves' disease",
    "parathyroid adenoma", "hyperparathyroidism", "hypoparathyroidism",
    "osteomalacia", "rickets", "Paget's disease of bone",
    "osteogenesis imperfecta", "achondroplasia", "scoliosis",
    "kyphosis", "lordosis", "spondylolisthesis", "spinal stenosis",
    "herniated disc", "sciatica", "carpal tunnel syndrome",
    "tendinitis", "bursitis", "plantar fasciitis", "gout",
    "pseudogout", "ankylosing spondylitis", "reactive arthritis",
    "psoriatic arthritis", "rheumatoid arthritis", "osteoarthritis",
    "fibromyalgia", "lupus", "scleroderma", "Sjogren's syndrome",
    "polymyalgia rheumatica", "vasculitis", "granulomatosis with polyangiitis",
    "microscopic polyangiitis", "Churg-Strauss syndrome",
    "Behcet's disease", "Kawasaki disease", "Takayasu's arteritis",
    "giant cell arteritis", "polyarteritis nodosa", "Henoch-Schonlein purpura",
    "Goodpasture's syndrome", "Wegener's granulomatosis",
    "eosinophilic granulomatosis with polyangiitis", "cryoglobulinemia",
    "amyloidosis", "sarcoidosis", "histiocytosis", "hemochromatosis",
    "Wilson's disease", "alpha-1 antitrypsin deficiency",
    "cystic fibrosis", "Gaucher's disease", "Niemann-Pick disease",
    "Tay-Sachs disease", "Fabry disease", "Hurler syndrome",
    "Hunter syndrome", "Sanfilippo syndrome", "Morquio syndrome",
    "Maroteaux-Lamy syndrome", "Sly syndrome", "metachromatic leukodystrophy",
    "adrenoleukodystrophy", "Krabbe disease", "Canavan disease",
    "Alexander disease", "Pelizaeus-Merzbacher disease",
    "Rett syndrome", "Angelman syndrome", "Prader-Willi syndrome",
    "fragile X syndrome", "Down syndrome", "Turner syndrome",
    "Klinefelter syndrome", "Noonan syndrome", "Williams syndrome",
    "Marfan syndrome", "Ehlers-Danlos syndrome", "osteogenesis imperfecta",
    "achondroplasia", "neurofibromatosis", "tuberous sclerosis",
    "von Hippel-Lindau disease", "Sturge-Weber syndrome",
    "ataxia-telangiectasia", "Fanconi anemia", "xeroderma pigmentosum",
    "Bloom syndrome", "Cockayne syndrome", "trichothiodystrophy",
    "progeria", "Werner syndrome", "Hutchinson-Gilford progeria syndrome",
    "myotonic dystrophy", "facioscapulohumeral muscular dystrophy",
    "Duchenne muscular dystrophy", "Becker muscular dystrophy",
    "limb-girdle muscular dystrophy", "Emery-Dreifuss muscular dystrophy",
    "oculopharyngeal muscular dystrophy", "congenital muscular dystrophy",
    "spinal muscular atrophy", "amyotrophic lateral sclerosis",
    "primary lateral sclerosis", "progressive muscular atrophy",
    "Kennedy's disease", "hereditary spastic paraplegia",
    "Friedreich's ataxia", "ataxia with oculomotor apraxia",
    "ataxia with vitamin E deficiency", "ataxia with isolated vitamin E deficiency",
    "ataxia with coenzyme Q10 deficiency", "ataxia with oculomotor apraxia type 1",
    "ataxia with oculomotor apraxia type 2", "ataxia-telangiectasia-like disorder",
    "spinocerebellar ataxia", "episodic ataxia", "dystonia",
    "Parkinson's disease", "progressive supranuclear palsy",
    "multiple system atrophy", "corticobasal degeneration",
    "Lewy body dementia", "frontotemporal dementia",
    "Alzheimer's disease", "vascular dementia", "Creutzfeldt-Jakob disease",
    "fatal familial insomnia", "Gerstmann-Straussler-Scheinker syndrome",
    "kuru", "chronic traumatic encephalopathy", "Huntington's disease",
    "prion disease"
]

def analyze_symptom(user_input):
    try:
        # Step 2.1: Input Processing with Hugging Face Model
        result = classifier(user_input, candidate_labels=categories)
        disease = result['labels'][0]
        print(f"Extracted disease/symptom: {disease}")

        # Step 2.2: Fetch Relevant YouTube Videos
        YOUTUBE_API_KEY = "YOUTUBE_API_KEY"  # Replace with your YouTube API key
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Search for videos related to the extracted disease/symptom
        request = youtube.search().list(
            q=disease,
            part="snippet",
            type="video",
            maxResults=25 # Fetch at least 12 videos
        )
        response = request.execute()

        # Extract video data (ID, thumbnail, title)
        video_data = [
            {
                "id": item['id']['videoId'],
                "thumbnail": item['snippet']['thumbnails']['default']['url'],
                "title": item['snippet']['title']
            }
            for item in response['items']
        ]
        print(f"Fetched {len(video_data)} videos.")

        # Step 2.3: Extract Video Transcripts
        transcripts = []
        for video in video_data:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video['id'])
                transcripts.append({
                    "id": video['id'],
                    "thumbnail": video['thumbnail'],
                    "title": video['title'],
                    "transcript": " ".join([t['text'] for t in transcript])
                })
            except:
                print(f"Transcript not available for video: {video['id']}")
        print(f"Fetched {len(transcripts)} transcripts.")

        # Step 2.4: Summarization with Hugging Face Model
        if not transcripts:
            return "No transcripts available for the selected videos. Please try another symptom."

        summaries = []
        for transcript in transcripts:
            if len(transcript['transcript']) > 1024:
                transcript['transcript'] = transcript['transcript'][:1024]  # Truncate if too long
            if len(transcript['transcript']) < 30:
                continue  # Skip if too short
            try:
                # Summarize the transcript
                summary = summarizer(transcript['transcript'], max_length=130, min_length=30, do_sample=False)
                # Extract prevention tips
                prevention = summarizer(transcript['transcript'], max_length=100, min_length=20, do_sample=False)
                # Extract cure/treatment information
                cure = summarizer(transcript['transcript'], max_length=100, min_length=20, do_sample=False)
                summaries.append({
                    "thumbnail": transcript['thumbnail'],
                    "title": transcript['title'],
                    "summary": summary[0]['summary_text'],
                    "prevention": prevention[0]['summary_text'],
                    "cure": cure[0]['summary_text'],
                    "video_id": transcript['id']
                })
            except Exception as e:
                print(f"Error summarizing transcript: {e}")

        if not summaries:
            return "No valid summaries could be generated. Please try another symptom."

        # Step 2.5: Combine Summaries with Thumbnails, Prevention, and Cure
        final_output = ""
        for i, summary in enumerate(summaries):
            final_output += f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 10px;">
                <img src="{summary['thumbnail']}" alt="Thumbnail" style="width: 120px; height: 90px; margin-right: 10px; border-radius: 5px;">
                <div style="flex: 1;">
                    <h3 style="margin: 0; font-size: 18px;">Video {i+1}: {summary['title']}</h3>
                    <p style="margin: 5px 0; font-size: 14px;"><strong>Summary:</strong> {summary['summary']}</p>
                    <details style="margin: 5px 0;">
                        <summary style="font-size: 14px; font-weight: bold; cursor: pointer;">🛡 Prevention Tips</summary>
                        <p style="margin: 5px 0; font-size: 14px;">{summary['prevention']}</p>
                    </details>
                    <details style="margin: 5px 0;">
                        <summary style="font-size: 14px; font-weight: bold; cursor: pointer;">💊 Cure/Treatment</summary>
                        <p style="margin: 5px 0; font-size: 14px;">{summary['cure']}</p>
                    </details>
                    <iframe width="300" height="200" src="https://www.youtube.com/embed/{summary['video_id']}" frameborder="0" allowfullscreen style="border-radius: 5px;"></iframe>
                </div>
            </div>
            """
        return final_output

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Step 3: Create a Gradio Interface
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""
    <div style='text-align: center;'>
        <h1>Welcome to the Symptom Analyzer! 🩺</h1>
        <p>Enter your symptom or condition, and the system will analyze relevant YouTube videos and provide a summary.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                lines=2,
                placeholder="Describe your symptom or condition...",
                label="Your Symptom"
            )
            examples = gr.Examples(
                examples=[
                    "I am having a headache",
                    "I feel dizzy and nauseous",
                    "My stomach hurts after eating",
                    "I have a migraine"
                ],
                inputs=user_input
            )
        with gr.Column():
            output = gr.HTML(label="Video Summaries with Thumbnails, Prevention, and Cure")

    analyze_button = gr.Button("Analyze", variant="primary")
    analyze_button.click(fn=analyze_symptom, inputs=user_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
