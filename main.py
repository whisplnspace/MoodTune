import torch
import torch.nn as nn
import librosa
import streamlit as st
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline

# Load NLP Models
tokenizer_text = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased")

# GPT-2 Model for Joke & Quote Generation
tokenizer_gen = AutoTokenizer.from_pretrained("gpt2-medium")
gen_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


def extract_mfcc(audio_path, sr=22050, n_mfcc=40):
    """Extracts MFCC features from an audio file and normalizes them."""
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

    # Normalize & take mean along time dimension
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.mean(dim=0).unsqueeze(0)  # Shape: (1, 40)


class MultiModalTransformer(nn.Module):
    """Model to combine text and audio embeddings."""

    def __init__(self, text_dim=768, audio_dim=40, hidden_dim=512):
        super(MultiModalTransformer, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_embeds, audio_embeds):
        text_embeds = self.text_proj(text_embeds)
        audio_embeds = self.audio_proj(audio_embeds)
        combined_embeds = torch.cat((text_embeds, audio_embeds), dim=-1)
        return self.fc_out(combined_embeds)


def interpret_mood(text):
    """Determines mood using sentiment analysis."""
    sentiment = sentiment_analyzer(text)[0]
    label = sentiment["label"]
    score = sentiment["score"]

    if label == "POSITIVE" and score > 0.75:
        return "Happy 😊"
    elif label == "NEGATIVE" and score > 0.75:
        return "Sad 😞"
    else:
        return "Neutral 😐"


def generate_text(mood_category, output_type="joke"):
    """Generates a joke or motivational quote based on mood."""
    prompt = f"Tell me a funny joke that fits the mood: {mood_category}." if output_type == "joke" else f"Give me an inspirational quote for someone feeling {mood_category}."

    input_ids = tokenizer_gen.encode(prompt, return_tensors="pt", truncation=True, max_length=50)

    output = gen_model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        num_beams=5,
        repetition_penalty=1.5,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer_gen.eos_token_id
    )

    return tokenizer_gen.decode(output[0], skip_special_tokens=True)


def main():
    st.title("MoodTune 🎵 - AI Mood-Based Joke & Quote Generator")
    st.markdown("#### 🌟 Describe your mood and upload an audio file!")

    text = st.text_input("Describe your mood:")
    audio_file = st.file_uploader("Upload an audio file:", type=["wav", "mp3"])
    output_type = st.radio("Choose Output Type:", ["Joke", "Inspirational Quote"])

    if st.button("Generate"):
        if text and audio_file:
            text_tokens = tokenizer_text(text, return_tensors="pt", padding=True, truncation=True)
            text_embedding = text_model(**text_tokens).last_hidden_state[:, 0, :]

            audio_embedding = extract_mfcc(audio_file)

            model = MultiModalTransformer()
            mood_embedding = model(text_embedding, audio_embedding)

            mood_category = interpret_mood(text)
            generated_text = generate_text(mood_category, output_type.lower())

            st.subheader(f"🧠 Mood Interpretation: {mood_category}")
            st.subheader("🎭 AI Generated Response:")
            st.success(generated_text)

            print(f"Input Mood: {text} | Mood: {mood_category}")
            print("Generated Text:", generated_text)

        else:
            st.warning("⚠️ Please enter text and upload an audio file!")


if __name__ == "__main__":
    main()
