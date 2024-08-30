import streamlit as st
import time
import openai
import json
from PyPDF2 import PdfReader
from fpdf import FPDF

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

def generate_pdf(questions):
    pdf = PDF()
    pdf.add_page()

    for i, q in enumerate(questions):
        question = f"Q{i+1}: {q['question']}"
        pdf.chapter_title(question)

        choices = "\n".join(q['choices'])
        pdf.chapter_body(choices)

        correct_answer = f"Correct answer: {q['correct_answer']}"
        pdf.chapter_body(correct_answer)

        explanation = f"Explanation: {q['explanation']}"
        pdf.chapter_body(explanation)

    return pdf.output(dest="S").encode("latin1")

def submit_answer(i, quiz_data):
    user_choice = st.session_state.get(f"user_choice_{i}")
    
    st.session_state.answers[i] = user_choice

    if user_choice == quiz_data['correct_answer']:
        st.session_state.feedback[i] = ("Correct", quiz_data.get('explanation', 'No explanation available'))
        st.session_state.correct_answers += 1
    else:
        st.session_state.feedback[i] = ("Incorrect", quiz_data.get('explanation', 'No explanation available'), quiz_data['correct_answer'])

def mc_quiz_app():
    st.subheader('Single Choice Exam')
    st.write('Please select one answer for each question.')

    questions = st.session_state.generated_questions

    if questions:
        if 'answers' not in st.session_state:
            st.session_state.answers = [None] * len(questions)
            st.session_state.feedback = [None] * len(questions)
            st.session_state.correct_answers = 0

        for i, quiz_data in enumerate(questions):
            st.markdown(f"### Frage {i+1}: {quiz_data['question']}")

            if st.session_state.answers[i] is None:
                user_choice = st.radio("Choose one answer:", quiz_data['choices'], key=f"user_choice_{i}")
                st.button(f"Submit your answer {i+1}", key=f"submit_{i}", on_click=submit_answer, args=(i, quiz_data))
            else:
                st.radio("Choose one answer:", quiz_data['choices'], key=f"user_choice_{i}", index=quiz_data['choices'].index(st.session_state.answers[i]), disabled=True)
                
                feedback_type = st.session_state.feedback[i][0]
                if feedback_type == "Correct":
                    st.success(st.session_state.feedback[i][0])
                else:
                    st.error(f"{st.session_state.feedback[i][0]} - Correct answer: {st.session_state.feedback[i][2]}")
                
                st.markdown(f"Explanation: {st.session_state.feedback[i][1]}")

        if all(answer is not None for answer in st.session_state.answers):
            score = st.session_state.correct_answers
            total_questions = len(questions)
            st.write(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh;">
                    <h1 style="font-size: 3em; color: gold;">🏆</h1>
                    <h1>Your Score: {score}/{total_questions}</h1>
                </div>
            """, unsafe_allow_html=True)

def download_pdf_app():
    st.subheader('Download Your Exam as PDF')

    questions = st.session_state.generated_questions

    if questions:
        for i, q in enumerate(questions):
            st.markdown(f"### Q{i+1}: {q['question']}")
            for choice in q['choices']:
                st.write(choice)
            st.write(f"**Correct answer:** {q['correct_answer']}")
            st.write(f"**Explanation:** {q['explanation']}")
            st.write("---")

        pdf_bytes = generate_pdf(questions)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="generated_exam.pdf",
            mime="application/pdf"
        )

def pdf_upload_app():
    st.subheader("Upload Your Content - Create Your Test Exam")
    st.write("Upload the content and we take care of the rest")

    content_text = ""
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        content_text += pdf_text
        st.success("PDF content added to the session.")
        
        # Display a sample of the extracted text for verification
        st.subheader("Sample of extracted PDF content:")
        st.text(content_text[:500] + "...")  # Display first 500 characters

        st.info("Generating the exam from the uploaded content. It will take just a minute...")
        chunks = chunk_text(content_text)
        questions = []
        for chunk in chunks:
            response, error = generate_mc_questions(chunk)
            if error:
                st.error(f"Error generating questions: {error}")
                break
            parsed_questions, parse_error = parse_generated_questions(response)
            if parse_error:
                st.error(parse_error)
                st.text("Full response:")
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
            st.success(f"The exam has been successfully created with {len(questions)} questions! Switch the Sidebar Panel to take the exam.")
            
            # Display a sample question for verification
            st.subheader("Sample generated question:")
            st.json(questions[0])
            
            # Instead of using st.rerun(), we'll set a flag in the session state
            st.session_state.should_change_mode = True
        else:
            st.error("No questions were generated. Please check the error messages above and try again.")
    else:
        st.warning("Please upload a PDF to generate the interactive exam.")

def main():
    st.title("Exam Creator")
    
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Upload PDF & Generate Questions"
    
    app_mode_options = ["Upload PDF & Generate Questions", "Take the Quiz", "Download as PDF"]
    
    # Check if we should change the mode
    if 'should_change_mode' in st.session_state and st.session_state.should_change_mode:
        st.session_state.app_mode = "Take the Quiz"
        st.session_state.should_change_mode = False  # Reset the flag
    
    st.session_state.app_mode = st.sidebar.selectbox("Choose the app mode", app_mode_options, index=app_mode_options.index(st.session_state.app_mode))
    
    if st.session_state.app_mode == "Upload PDF & Generate Questions":
        pdf_upload_app()
    elif st.session_state.app_mode == "Take the Quiz":
        if 'mc_test_generated' in st.session_state and st.session_state.mc_test_generated:
            if 'generated_questions' in st.session_state and st.session_state.generated_questions:
                mc_quiz_app()
            else:
                st.warning("No generated questions found. Please upload a PDF and generate questions first.")
        else:
            st.warning("Please upload a PDF and generate questions first.")
    elif st.session_state.app_mode == "Download as PDF":
        download_pdf_app()

if __name__ == '__main__':
    main()
