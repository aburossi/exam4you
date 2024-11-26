import streamlit as st
import time
import openai
import json
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import io

# Set Streamlit page configuration
st.set_page_config(page_title="Exam Creator", page_icon="📝", layout="wide")

__version__ = "1.3.0"

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
    Processes an uploaded image file and returns it as a PIL Image object.
    """
    try:
        image = Image.open(file)
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def process_uploaded_file(uploaded_file):
    """
    Processes an uploaded file based on its type.
    Returns a dictionary with the file type and content.
    """
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
        if text:
            return {"type": "text", "content": text}
        else:
            st.error("Could not extract text from the PDF file.")
            return None

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
        if text:
            return {"type": "text", "content": text}
        else:
            st.error("Could not extract text from the DOCX file.")
            return None

    elif file_type.startswith("image/"):
        image = process_image(uploaded_file)
        if image:
            return {"type": "image", "content": image}
        else:
            st.error("Could not process the uploaded image.")
            return None

    else:
        st.error("Unsupported file type. Please upload a PDF, DOCX, or image file.")
        return None

# --------------------------- PDF Upload and Question Generation ---------------------------

def pdf_upload_app():
    """
    Handles file upload, content extraction, and question generation.
    """
    st.subheader("Upload Your File - Create Your Exam")
    st.write("Upload a PDF, DOCX, or Image, and we'll handle the rest.")

    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "jpg", "jpeg", "png"])
    if uploaded_file:
        processed_file = process_uploaded_file(uploaded_file)
        if processed_file:
            file_type = processed_file["type"]
            content = processed_file["content"]

            if file_type == "text":
                st.success("File content successfully extracted.")
                st.text_area("Extracted Text (Preview):", value=content[:500] + "...", height=200)

                # Continue with question generation as in the original implementation
                st.info("Generating exam questions from the uploaded content. This may take a minute...")
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
                        st.text_area("Full Response:", value=response, height=200)
                        break
                    if parsed_questions:
                        questions.extend(parsed_questions)
                        if len(questions) >= 20:
                            questions = questions[:20]  # Limit to 20 questions
                            break
                if questions:
                    st.session_state.generated_questions = questions
                    st.session_state.content_text = content
                    st.session_state.mc_test_generated = True
                    st.success(f"Exam successfully generated with {len(questions)} questions!")

                    # Display options to proceed
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Take the Exam"):
                            st.session_state.app_mode = "Take Exam"
                            st.experimental_rerun()
                    with col2:
                        if st.button("Download Exam"):
                            st.session_state.app_mode = "Download Exam"
                            st.experimental_rerun()
                else:
                    st.error("No questions were generated. Please check the above error messages and try again.")

            elif file_type == "image":
                st.image(content, caption="Uploaded Image", use_column_width=True)
                st.warning("Question generation from images is not supported in this version.")
    else:
        st.warning("Please upload a file to generate an interactive exam.")

# --------------------------- Main Application ---------------------------

def main():
    """
    The main function that controls the Streamlit app's flow.
    """
    st.title("📝 Exam Creator")
    st.markdown(f"**Version:** {__version__}")

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Upload PDF & Generate Questions"

    # Define app modes
    app_mode_options = ["Upload PDF & Generate Questions", "Take Exam", "Download Exam"]
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.session_state.app_mode = st.sidebar.selectbox("Select App Mode", app_mode_options, index=app_mode_options.index(st.session_state.app_mode))

    # Render the selected app mode
    if st.session_state.app_mode == "Upload PDF & Generate Questions":
        pdf_upload_app()
    elif st.session_state.app_mode == "Take Exam":
        if 'mc_test_generated' in st.session_state and st.session_state.mc_test_generated:
            if 'generated_questions' in st.session_state and st.session_state.generated_questions:
                mc_quiz_app()
            else:
                st.warning("No generated questions found. Please upload a file and generate questions first.")
        else:
            st.warning("Please upload a file and generate questions first.")
    elif st.session_state.app_mode == "Download Exam":
        if 'mc_test_generated' in st.session_state and st.session_state.mc_test_generated:
            download_files_app()
        else:
            st.warning("Please upload a file and generate questions first.")

if __name__ == '__main__':
    main()
