#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from groq import Groq
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import cv2
from datetime import datetime


def query_groq(sys_content, user_content, model):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content


class ElderChatbot:
    def __init__(self):
        self.conversation_history = []
        self.user_name = None
        self.user_image = None

    def capture_selfie(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Could not access camera"

        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"selfie_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.user_image = filename

        cap.release()
        return filename

    def get_bot_response(self, user_input):
        system_prompt = """You are a friendly and patient chatbot speaking with an elderly person. 
        Show genuine interest in their stories and experiences. Ask follow-up questions that encourage 
        them to share more details about their life stories."""

        response = query_groq(system_prompt, user_input, model="llama-3.1-8b-instant")
        self.conversation_history.append({"user": user_input, "bot": response})
        return response

    def generate_blog_post(self):
        system_prompt = """Create a compelling blog post from the following conversation. 
        The blog post should:
        1. Have an engaging title
        2. Be structured in clear sections
        3. Focus on the most interesting life stories and insights shared
        4. Include direct quotes when relevant
        5. Have a thoughtful conclusion
        6. Be between 500-1000 words
        Format the response with the title on top, followed by the content in paragraphs."""

        conversation_text = str(self.conversation_history)
        blog_post = query_groq(system_prompt, conversation_text, model="llama-3.1-70b-versatile")
        self.save_as_pdf(blog_post)
        return blog_post

    def save_as_pdf(self, blog_content):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elder_story_{timestamp}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create custom style for the title
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Heading1"], fontSize=24, spaceAfter=30
        )

        # Split content into title and body
        content_parts = blog_content.split("\n", 1)
        title = content_parts[0]
        body = content_parts[1] if len(content_parts) > 1 else ""

        # Prepare the story elements
        story = []

        # Add title
        story.append(Paragraph(title, title_style))

        # Add image if available
        if self.user_image and os.path.exists(self.user_image):
            img = Image(self.user_image, width=300, height=300)
            story.append(img)
        story.append(Spacer(1, 20))

        # Add blog content
        for paragraph in body.split("\n\n"):
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles["Normal"]))
                story.append(Spacer(1, 12))

        # Generate PDF
        doc.build(story)
        return filename


def main():
    chatbot = ElderChatbot()
    print("Elder Chatbot: Hello! I'd love to chat with you. What's your name?")

    while True:
        user_input = input("You (type 'selfie' for photo, 'quit' to end): ")

        if user_input.lower() == "selfie":
            result = chatbot.capture_selfie()
            print(f"Selfie captured: {result}")
            continue

        elif user_input.lower() == "quit":
            print("\nGenerating your story...")
            blog_post = chatbot.generate_blog_post()
            print("\nYour story has been saved as a PDF!")
            break

        response = chatbot.get_bot_response(user_input)
        print(f"Elder Chatbot: {response}")


if __name__ == "__main__":
    main()
