import streamlit as st
import time
import openai
import json
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(page_title="Exam Creator", page_icon="📝", layout="wide")

__version__ = "1.3.0"

# --------------------------- Helper Functions ---------------------------

def stream_llm_response(messages, model_params):
    """
    Sends messages to OpenAI's ChatCompletion API and returns the response content.
    """
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model=model_params.get("model", "gpt-4o-mini"),
        messages=messages,
        temperature=model_params.get("temperature", 0.5),
        max_tokens=13000,
    )
    return response.choices[0].message['content']

def extract_text_from_pdf(pdf_file):
    """
    Extracts text content from an uploaded PDF file.
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text

def extract_text_from_docx(docx_file):
    """
    Extracts text content from an uploaded DOCX file.
    """
    document = Document(docx_file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)

def process_image_file(image_file):
    """
    Processes an uploaded image file using PIL.
    """
    try:
        image = Image.open(image_file)
        return image
    except Exception as e:
        st.error(f"Error processing image file: {e}")
        return None

def chunk_text(text, max_tokens=3000):
    """
    Splits the extracted text into manageable chunks based on the maximum token limit.
    """
    sentences = text.split('. ')
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_tokens:
            chunks.append(chunk)
            chunk = sentence + ". "
        else:
            chunk += sentence + ". "
    if chunk:
        chunks.append(chunk)
    return chunks

def generate_mc_questions(content_text):
    """
    Generates multiple-choice questions based on the provided content using OpenAI's API.
    """
    system_prompt = (
        "Sie sind ein Lehrer für Allgemeinbildung und sollen eine Prüfung zum Thema des eingereichten Inhalts erstellen. "
        "Erstellen Sie eine Single-Choice-Prüfung auf Oberstufenniveau. "
        "Jede Frage soll genau eine richtige Antwort haben. "
        "Erstellen Sie so viele Prüfungsfragen wie nötig, um den gesamten Inhalt abzudecken, aber maximal 20 Fragen. "
        "Geben Sie die Ausgabe im JSON-Format an. "
        "Das JSON sollte folgende Struktur haben: [{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}, ...]. "
        "Stellen Sie sicher, dass das JSON gültig und korrekt formatiert ist."
    )
    user_prompt = f"Using the following content, create single-choice questions:\n\n{content_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = stream_llm_response(messages, model_params={"model": "gpt-4o-mini", "temperature": 0.5})
        return response, None
    except Exception as e:
        return None, str(e)

def parse_generated_questions(response):
    """
    Parses the JSON response from OpenAI into Python objects.
    """
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start == -1 or json_end == 0:
            return None, f"No JSON data found in the response. First 500 characters of response:\n{response[:500]}..."
        json_str = response[json_start:json_end]

        questions = json.loads(json_str)
        return questions, None
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}\n\nFirst 500 characters of response:\n{response[:500]}..."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}\n\nFirst 500 characters of response:\n{response[:500]}..."

# --------------------------- File Upload and Processing ---------------------------

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file and returns extracted text or image data.
    """
    if uploaded_file.type == "application/pdf":
        return {"type": "text", "content": extract_text_from_pdf(uploaded_file)}
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return {"type": "text", "content": extract_text_from_docx(uploaded_file)}
    elif uploaded_file.type.startswith("image/"):
        return {"type": "image", "content": process_image_file(uploaded_file)}
    else:
        st.error("Unsupported file type. Please upload a PDF, DOCX, or Image file.")
        return None

def handle_uploaded_file():
    """
    Handles file uploads and displays content for further processing.
    """
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or Image file", type=["pdf", "docx", "jpg", "jpeg", "png"])
    if uploaded_file:
        file_data = process_uploaded_file(uploaded_file)
        if file_data:
            if file_data["type"] == "text":
                st.text_area("Extracted Content:", value=file_data["content"], height=300)
                return file_data["content"]
            elif file_data["type"] == "image":
                st.image(file_data["content"], caption="Uploaded Image", use_column_width=True)
                st.warning("Question generation from images is not yet implemented.")
        else:
            st.error("Failed to process the uploaded file.")
    return None

# --------------------------- Quiz Interaction and Download ---------------------------

def take_exam(questions):
    """
    Allows the user to take the generated exam.
    """
    st.subheader("Take the Exam")
    for i, question in enumerate(questions):
        st.write(f"**Q{i+1}:** {question['question']}")
        user_answer = st.radio(f"Choose your answer for Q{i+1}:", question['choices'], key=f"answer_{i}")
        if st.button(f"Submit Answer for Q{i+1}", key=f"submit_{i}"):
            if user_answer == question['correct_answer']:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is: {question['correct_answer']}")

def download_exam(questions):
    """
    Provides options to download the generated exam.
    """
    st.subheader("Download the Exam")
    format_choice = st.radio("Select format to download:", ["PDF", "DOCX"])
    include_answers = st.checkbox("Include answers and explanations")

    if st.button("Download"):
        if format_choice == "PDF":
            st.write("PDF download functionality not implemented in this version.")
        elif format_choice == "DOCX":
            st.write("DOCX download functionality not implemented in this version.")
        else:
            st.error("Unsupported format selected.")

# --------------------------- Main Application ---------------------------

def main():
    """
    The main function that controls the Streamlit app's flow.
    """
    st.title("📝 Exam Creator with File Handling")
    st.markdown(f"**Version:** {__version__}")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose App Mode", ["Upload & Generate", "Take Exam", "Download Exam"])

    if app_mode == "Upload & Generate":
        content = handle_uploaded_file()
        if content:
            chunks = chunk_text(content)
            questions = []
            for chunk in chunks:
                response, error = generate_mc_questions(chunk)
                if error:
                    st.error(f"Error generating questions: {error}")
                    break
                parsed_questions, parse_error = parse_generated_questions(response)
                if parse_error:
                    st.error(parse_error)
                    break
                questions.extend(parsed_questions)
            if questions:
                st.session_state["questions"] = questions
                st.success(f"Generated {len(questions)} questions.")
    elif app_mode == "Take Exam":
        if "questions" in st.session_state:
            take_exam(st.session_state["questions"])
        else:
            st.warning("No questions generated. Upload a file and generate questions first.")
    elif app_mode == "Download Exam":
        if "questions" in st.session_state:
            download_exam(st.session_state["questions"])
        else:
            st.warning("No questions generated. Upload a file and generate questions first.")

if __name__ == '__main__':
    main()
