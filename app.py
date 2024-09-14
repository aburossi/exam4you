import streamlit as st
import time
import openai
import json
from PyPDF2 import PdfReader
from fpdf import FPDF
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


# Set page config
st.set_page_config(page_title="Exam Creator", page_icon="📝")

__version__ = "1.3.0"

# Main app functions
def stream_llm_response(messages, model_params):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
        model=model_params["model"] if "model" in model_params else "gpt-4o-mini",
        messages=messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.5,
        max_tokens=12000,
    )
    return response.choices[0].message['content']

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, max_tokens=3000):
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
    system_prompt = (
        "Sie sind ein Lehrer für Allgemeinbildung und sollen eine Prüfung zum Thema des eingereichten PDFs erstellen. "
        "Verwenden Sie den Inhalt des PDFs (bitte gründlich analysieren) und erstellen Sie eine Single-Choice-Prüfung auf Oberstufenniveau. "
        "Jede Frage soll genau eine richtige Antwort haben. "
        "Erstellen Sie so viele Prüfungsfragen, wie nötig sind, um den gesamten Inhalt abzudecken, aber maximal 20 Fragen. "
        "Geben Sie die Ausgabe im JSON-Format an. "
        "Das JSON sollte folgende Struktur haben: [{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}, ...]. "
        "Stellen Sie sicher, dass das JSON gültig und korrekt formatiert ist."
    )
    user_prompt = (
        "Using the following content from the uploaded PDF, create single-choice questions. "
        "Ensure that each question is based on the information provided in the PDF content and has exactly one correct answer. "
        "Create as many questions as necessary to cover the entire content, but no more than 20 questions. "
        "Provide the output in JSON format with the following structure: "
        "[{'question': '...', 'choices': ['...'], 'correct_answer': '...', 'explanation': '...'}, ...]. "
        "Ensure the JSON is valid and properly formatted.\n\nPDF Content:\n\n"
    ) + content_text

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

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Generated Exam', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, title)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def print_checkbox(self, x, y, checked=False):
        """
        Draws a checkbox at the specified (x, y) coordinates.
        If `checked` is True, the checkbox will be marked.
        """
        # Draw the square for the checkbox
        self.rect(x, y, 5, 5)
        
        if checked:
            # Draw a check mark
            self.set_line_width(0.5)
            self.line(x, y, x + 5, y + 5)
            self.line(x + 5, y, x, y + 5)
            self.set_line_width(0.2)  # Reset to default

def generate_docx(questions, include_answers=True):
    document = Document()
    
    # Set document title
    title = document.add_heading('Generated Exam', level=1)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    
    for i, q in enumerate(questions):
        # Add question number and text
        question_text = f"Q{i+1}: {q['question']}"
        document.add_paragraph(question_text, style='List Number')
        
        # Add choices
        for choice in q['choices']:
            document.add_paragraph(choice, style='List Bullet')
        
        if include_answers:
            # Add correct answer
            correct_answer = f"**Correct Answer:** {q['correct_answer']}"
            p = document.add_paragraph()
            run = p.add_run(correct_answer)
            run.bold = True
            
            # Add explanation
            explanation = f"**Explanation:** {q['explanation']}"
            p = document.add_paragraph()
            run = p.add_run(explanation)
            run.italic = True
            
            # Add "Test on paper" checkbox
            p = document.add_paragraph()
            p.add_run("[ ] Test on paper")
        
        # Add a horizontal line for separation
        document.add_paragraph().add_run().add_break()
    
    # Save the document to a BytesIO object
    from io import BytesIO
    docx_io = BytesIO()
    document.save(docx_io)
    docx_io.seek(0)
    
    return docx_io.getvalue()


def generate_pdf(questions, include_answers=True):
    pdf = PDF()
    pdf.add_page()

    for i, q in enumerate(questions):
        question = f"Q{i+1}: {q['question']}"
        pdf.chapter_title(question)

        # List the choices
        choices = "\n".join(q['choices'])
        pdf.chapter_body(choices)

        if include_answers:
            # Add the correct answer
            correct_answer = f"Correct answer: {q['correct_answer']}"
            pdf.chapter_body(correct_answer)

            # Add the explanation
            explanation = f"Explanation: {q['explanation']}"
            pdf.chapter_body(explanation)

            # Print checkbox for "Test on paper"
            current_y = pdf.get_y()  # Get current y position
            pdf.print_checkbox(10, current_y, True)  # Draw a checked checkbox
            pdf.set_xy(16, current_y)  # Move to the right of the checkbox
            pdf.cell(0, 5, "Test on paper")
            pdf.ln()

    return pdf.output(dest="S").encode("latin1")



def submit_answer(i, quiz_data):
    user_choice = st.session_state.get(f"user_choice_{i}")
    
    st.session_state.answers[i] = user_choice

    if user_choice == quiz_data['correct_answer']:
        st.session_state.feedback[i] = ("Richtig", quiz_data.get('explanation', 'Keine Erklärung verfügbar'))
        st.session_state.correct_answers += 1
    else:
        st.session_state.feedback[i] = ("Falsch", quiz_data.get('explanation', 'Keine Erklärung verfügbar'), quiz_data['correct_answer'])

def mc_quiz_app():
    st.subheader('Single-Choice-Prüfung')
    st.write('Bitte wählen Sie eine Antwort für jede Frage.')

    questions = st.session_state.generated_questions

    if questions:
        if 'answers' not in st.session_state:
            st.session_state.answers = [None] * len(questions)
            st.session_state.feedback = [None] * len(questions)
            st.session_state.correct_answers = 0

        for i, quiz_data in enumerate(questions):
            st.markdown(f"### Frage {i+1}: {quiz_data['question']}")

            if st.session_state.answers[i] is None:
                user_choice = st.radio("Wählen Sie die richtige Antwort:", quiz_data['choices'], key=f"user_choice_{i}")
                st.button(f"Antwort {i+1} überprüfen", key=f"submit_{i}", on_click=submit_answer, args=(i, quiz_data))
            else:
                st.radio("Wählen Sie eine Antwort:", quiz_data['choices'], key=f"user_choice_{i}", index=quiz_data['choices'].index(st.session_state.answers[i]), disabled=True)
                
                feedback_type = st.session_state.feedback[i][0]
                if feedback_type == "Richtig":
                    st.success(st.session_state.feedback[i][0])
                else:
                    st.error(f"{st.session_state.feedback[i][0]} - Richtige Antwort: {st.session_state.feedback[i][2]}")
                
                st.markdown(f"Erklärung: {st.session_state.feedback[i][1]}")

        if all(answer is not None for answer in st.session_state.answers):
            score = st.session_state.correct_answers
            total_questions = len(questions)
            st.write(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh;">
                    <h1 style="font-size: 3em; color: gold;">🏆</h1>
                    <h1>Ihr Ergebnis: {score}/{total_questions}</h1>
                </div>
            """, unsafe_allow_html=True)

def download_files_app():
    st.subheader('Prüfung herunterladen')
    
    questions = st.session_state.generated_questions
    
    if questions:
        # Display questions preview
        for i, q in enumerate(questions):
            st.markdown(f"### Frage {i+1}: {q['question']}")
            for choice in q['choices']:
                st.write(choice)
            if 'correct_answer' in q:
                st.write(f"**Richtige Antwort:** {q['correct_answer']}")
            if 'explanation' in q:
                st.write(f"**Erklärung:** {q['explanation']}")
            st.write("---")
    
        # Choose format and inclusion of answers
        doc_type = st.radio("Wählen Sie die Ausgabe:", ["DOCX mit Lösungen", "DOCX ohne Lösungen"])
        
        if st.button("Datei generieren"):
            if doc_type == "DOCX mit Lösungen":
                file_bytes = generate_docx(questions, include_answers=True)
                file_name = "prüfung_mit_antworten.docx"
            else:
                file_bytes = generate_docx(questions, include_answers=False)
                file_name = "prüfung_ohne_antworten.docx"
            
            st.download_button(
                label="DOCX herunterladen",
                data=file_bytes,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

def pdf_upload_app():
    st.subheader("Laden Sie Ihren Inhalt hoch - Erstellen Sie Ihre Testprüfung")
    st.write("Laden Sie den Inhalt hoch und wir kümmern uns um den Rest")

    content_text = ""
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    uploaded_pdf = st.file_uploader("Laden Sie ein PDF-Dokument hoch", type=["pdf"])
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        content_text += pdf_text
        st.success("PDF-Inhalt zur Sitzung hinzugefügt.")
        
        # Display a sample of the extracted text for verification
        st.subheader("Beispiel des extrahierten PDF-Inhalts:")
        st.text(content_text[:500] + "...")  # Display first 500 characters

        st.info("Ich erstelle die Prüfung aus den hochgeladenen Inhalten. Dies dauert nur eine Minute.....")
        chunks = chunk_text(content_text)
        questions = []
        for chunk in chunks:
            response, error = generate_mc_questions(chunk)
            if error:
                st.error(f"Fehler beim Generieren der Fragen: {error}")
                break
            parsed_questions, parse_error = parse_generated_questions(response)
            if parse_error:
                st.error(parse_error)
                st.text("Vollständige Antwort:")
                st.text(response)
                break
            if parsed_questions:
                questions.extend(parsed_questions)
                if len(questions) >= 20:
                    questions = questions[:20]  # Limit to 20 questions
                    break
        if questions:
            st.session_state.generated_questions = questions
            st.session_state.content_text = content_text
            st.session_state.mc_test_generated = True
            st.success(f"Die Prüfung wurde erfolgreich mit {len(questions)} Fragen erstellt!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Prüfung ablegen"):
                    st.session_state.app_mode = "Prüfung ablegen"
                    st.experimental_rerun()
            with col2:
                if st.button("Als PDF herunterladen"):
                    st.session_state.app_mode = "Als PDF herunterladen"
                    st.experimental_rerun()
        else:
            st.error("Es wurden keine Fragen generiert. Bitte überprüfen Sie die obigen Fehlermeldungen und versuchen Sie es erneut.")
    else:
        st.warning("Bitte laden Sie ein PDF hoch, um die interaktive Prüfung zu generieren.")

def main():
    st.title("Prüfungsersteller")
    
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "PDF hochladen & Fragen generieren"
    
    app_mode_options = ["PDF hochladen & Fragen generieren", "Prüfung ablegen", "Prüfung herunterladen"]
    
    st.session_state.app_mode = st.sidebar.selectbox("Wählen Sie den App-Modus", app_mode_options, index=app_mode_options.index(st.session_state.app_mode))
    
    if st.session_state.app_mode == "PDF hochladen & Fragen generieren":
        pdf_upload_app()
    elif st.session_state.app_mode == "Prüfung ablegen":
        if 'mc_test_generated' in st.session_state and st.session_state.mc_test_generated:
            if 'generated_questions' in st.session_state and st.session_state.generated_questions:
                mc_quiz_app()
            else:
                st.warning("Keine generierten Fragen gefunden. Bitte laden Sie zuerst ein PDF hoch und generieren Sie Fragen.")
        else:
            st.warning("Bitte laden Sie zuerst ein PDF hoch und generieren Sie Fragen.")
    elif st.session_state.app_mode == "Prüfung herunterladen":
        download_docx_app()  # Use download_files_app() if offering multiple formats

if __name__ == '__main__':
    main()

