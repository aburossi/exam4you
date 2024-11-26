import streamlit as st
import time
import openai
import json
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import io
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="Exam Creator", page_icon="📝", layout="wide")

__version__ = "1.3.0"

# --------------------------- System Prompt ---------------------------

system_prompt = (
    "Sie sind ein Lehrer für Allgemeinbildung und sollen eine Prüfung zum Thema des eingereichten Inhalts erstellen. "
    "Verwenden Sie den Inhalt (bitte gründlich analysieren) und erstellen Sie eine Single-Choice-Prüfung auf Oberstufenniveau. "
    "Jede Frage soll genau eine richtige Antwort haben. "
    "Erstellen Sie so viele Prüfungsfragen, wie nötig sind, um den gesamten Inhalt abzudecken, aber maximal 20 Fragen. "
    "Geben Sie die Ausgabe im JSON-Format an. "
    "Das JSON sollte folgende Struktur haben: [{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}, ...]. "
    "Stellen Sie sicher, dass das JSON gültig und korrekt formatiert ist."
)

# --------------------------- File Handling Functions ---------------------------

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    """
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """
    Extracts text from a DOCX file.
    """
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def process_image(file):
    """
    Processes an uploaded image, resizes it if necessary, and converts it to a Base64-encoded string.
    """
    try:
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        max_size = 1000
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --------------------------- Question Generation Functions ---------------------------

def generate_mc_questions(content, system_prompt):
    """
    Generates multiple-choice questions based on the provided content using OpenAI's API.
    """
    user_prompt = (
        "Using the following content from the uploaded document, create single-choice questions. "
        "Ensure that each question is based on the information provided in the document content and has exactly one correct answer. "
        "Create as many questions as necessary to cover the entire content, but no more than 20 questions. "
        "Provide the output in JSON format with the following structure: "
        "[{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}]. "
        "Ensure the JSON is valid and properly formatted.\n\nDocument Content:\n\n" + content
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message["content"], None
    except Exception as e:
        return None, f"Error generating questions: {e}"

def get_questions_from_image(image, system_prompt, user_prompt):
    """
    Generates questions based on an image using OpenAI's API.
    """
    try:
        base64_image = process_image(image)
        if not base64_image:
            return None, "Image processing failed."

        image_payload = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low",
            },
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}, image_payload],
            },
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message["content"], None
    except Exception as e:
        return None, f"Error during question generation: {e}"

# --------------------------- App Logic ---------------------------

def pdf_upload_app():
    """
    Handles file upload, content extraction, and question generation.
    """
    st.subheader("Upload Your File - Create Your Exam")
    st.write("Upload a PDF, DOCX, or Image, and we'll handle the rest.")

    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "jpg", "jpeg", "png"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
            file_type = "text"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = extract_text_from_docx(uploaded_file)
            file_type = "text"
        elif uploaded_file.type.startswith("image/"):
            content = process_image(uploaded_file)
            file_type = "image"
        else:
            st.error("Unsupported file type.")
            return

        if file_type == "text" and content:
            st.success("File content successfully extracted.")
            st.text_area("Extracted Text (Preview):", value=content[:500] + "...", height=200)

            st.info("Generating exam questions from the uploaded content. This may take a minute...")
            response, error = generate_mc_questions(content, system_prompt)
            if error:
                st.error(error)
            else:
                st.text_area("Generated Questions:", value=response, height=300)
                st.download_button(
                    label="Download Questions",
                    data=response,
                    file_name="questions_from_document.json",
                    mime="application/json",
                )
        elif file_type == "image" and content:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            user_prompt = (
                "Using the content derived from the uploaded image, create single-choice questions. "
                "Ensure that each question is based on the image content and has exactly one correct answer. "
                "Create as many questions as necessary to cover the entire content, but no more than 20 questions. "
                "Provide the output in JSON format with the following structure: "
                "[{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}]. "
                "Ensure the JSON is valid and properly formatted."
            )

            response, error = get_questions_from_image(uploaded_file, system_prompt, user_prompt)
            if error:
                st.error(error)
            else:
                st.text_area("Generated Questions:", value=response, height=300)
                st.download_button(
                    label="Download Questions",
                    data=response,
                    file_name="questions_from_image.json",
                    mime="application/json",
                )
        else:
            st.error("Could not process the uploaded file.")
    else:
        st.warning("Please upload a file to generate an interactive exam.")

def main():
    """
    Main function that controls the Streamlit app's flow.
    """
    st.title("📝 Exam Creator")
    st.markdown(f"**Version:** {__version__}")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["Upload PDF & Generate Questions"])
    
    if app_mode == "Upload PDF & Generate Questions":
        pdf_upload_app()

if __name__ == '__main__':
    main()
