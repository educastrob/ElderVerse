#!/usr/bin/env python3
# -*- coding: utf-8 -*- from groq import Groq import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import cv2
from datetime import datetime
from collections import deque
from groq import Groq
import os


def query_groq(messages, model):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=list(messages),  # Convert deque to list
        model=model,
    )
    return chat_completion.choices[0].message.content


class ElderChatbot:
    def __init__(self):
        self.max_messages = 15
        self.messages = deque(maxlen=self.max_messages)
        self.messages.append(
            {
                "role": "system",
                "content": """You are a friendly and patient chatbot speaking with an elderly person.
                Show genuine interest in their stories and experiences.""",
            }
        )
        self.user_name = None
        self.user_image = None

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_bot_response(self, user_input):
        self.add_message("user", user_input)
        response = query_groq(self.messages, model="llama-3.2-11b-text-preview")
        self.add_message("assistant", response)
        return response

    def generate_story(self):
        story_prompt = {
            "role": "system",
            "content": """Create a blog post from the following conversation. 
            The blog post should:
            1. Have an engaging title
            2. Be structured in clear sections
            3. Focus on the most interesting life stories and insights shared
            4. Include direct quotes when relevant
            5. Have a thoughtful conclusion
            6. Be between 500-1000 words
            7. Don't use markdown, because the final format will be a pdf.
            Format the response with the title on top, followed by the content in paragraphs.""",
        }

        # Create a temporary message list for story generation
        story_messages = [
            story_prompt,
            {"role": "user", "content": str(list(self.messages))},
        ]

        blog_post = query_groq(story_messages, model="llama-3.1-70b-versatile")
        self.save_as_pdf(blog_post)
        return blog_post

    def save_as_pdf(self, blog_content):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"elder_story_{timestamp}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Heading1"], fontSize=24, spaceAfter=30
        )

        content_parts = blog_content.split("\n", 1)
        title = content_parts[0]
        body = content_parts[1] if len(content_parts) > 1 else ""

        story = []
        story.append(Paragraph(title, title_style))

        if self.user_image and os.path.exists(self.user_image):
            img = Image(self.user_image, width=300, height=300)
            story.append(img)
        story.append(Spacer(1, 20))

        for paragraph in body.split("\n\n"):
            if paragraph.strip():
                story.append(Paragraph(paragraph, styles["Normal"]))
                story.append(Spacer(1, 12))

        doc.build(story)
        return filename


def main():
    chatbot = ElderChatbot()
    initial_message = "Hello! I'd love to chat with you. What's your name?"
    print(f"Elder Chatbot: {initial_message}")
    chatbot.add_message("assistant", initial_message)

    while True:
        try:
            user_input = input("type 'quit' to end): ")
        except EOFError:
            print("\nEOF detected. Exiting...")
            break

        if user_input.lower() == "quit":
            print("\nGenerating your story...")
            chatbot.generate_story()
            print("\nYour story has been saved as a PDF!")
            break

        response = chatbot.get_bot_response(user_input)
        print(f"Elder Chatbot: {response}")


if __name__ == "__main__":
    main()
