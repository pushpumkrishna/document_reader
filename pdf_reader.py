import os
from typing import List
import pdfx
import PyPDF2
from PyPDF2 import PdfReader
from transformers import pipeline


df = pdfx.PDFx("/Users/pushpum/Downloads/MACHINE LEARNING.pdf")
# print(df.get_text())
#
# df2 = PdfReader("/Users/pushpum/Downloads/MACHINE LEARNING.pdf")
# pdf_data = []

pdf_text_data = []


def read_pdf(location: str) -> List:
    """
    Function to read all pdf file in a location and return
    :param location: string
    :return: list of strings
    """

    for filename in os.listdir(location):
        if filename.endswith(".pdf"):
            file_path = os.path.join(location, filename)
            try:
                with open(file_path, "rb") as pdf_file:
                    pdf = PdfReader(pdf_file)
                    text = ""
                    for page_num in range(len(pdf.pages)):
                        page = pdf.pages[page_num]
                        text += page.extract_text()
                    pdf_text_data.append({"Filename": filename, "text": text})
                return pdf_text_data
            except Exception as e:
                print("Error reading pdf: ", e)
                continue


def extract_multiple_pdfs_text(
        pdf_directory: str
):

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)

            pdf_reader = PdfReader(file_path)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

            print(f"PDF: {file_path}\nText: {text[:100]}...\n{'*' * 100 }")


# Load the summarization pipeline
summarizer = pipeline("summarization")


# Function to generate summary for a PDF
def generate_summary(
        pdf_path,
        chunk_size=1000,
        max_length=100,
        min_length=50
):
    for filename in os.listdir(pdf_path):
        if filename.lower().endswith(".pdf"):
            try:
                with open(os.path.join(pdf_path, filename), "rb") as file:
                    pdf_text = ""
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        pdf_text = page.extract_text()
                    # Split text into chunks
                    chunks = [
                        pdf_text[i: i + chunk_size]
                        for i in range(0, len(pdf_text), chunk_size)
                    ]
                    # Generate summaries for each chunk

                    summaries = []
                    for i, chunk in enumerate(chunks):
                        summary = summarizer(
                            chunk,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False,
                        )
                summaries.append(summary[0]["summary_text"])
                print("File Name: {}\n{}".format(filename, summaries))
                # print("/n".join(summaries))
                # print(summaries)

            except Exception as e:
                print("Error reading pdf: ", e)

    return summaries


if __name__ == "__main__":
    print("process started !!")
    pdf_location = "./pdfs"
    # final_text = read_pdf(pdf_location)
    # print(final_text)

    # extract_multiple_pdfs_text(pdf_directory=pdf_location)

    a = generate_summary(pdf_location)
    print(a)
