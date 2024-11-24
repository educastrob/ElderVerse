#!/usr/bin/env python3
import dspy
from groq import Groq
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
from collections import deque
from typing import List
from dspy.teleprompt import BootstrapFewShot


# Configure DSPy with Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = "llama-3.1-70b-versatile"
dspy.configure(lm=dspy.LM(model, api_base=client, api_key="YOUR_GROQ_API_KEY"))


class ElderConversationSignature(dspy.Signature):
    """Defines the input/output signature for elderly conversation management"""

    context = dspy.InputField(desc="Previous conversation context")
    user_input = dspy.InputField(desc="Latest user message")
    should_ask_question = dspy.InputField(
        desc="Boolean indicating if should ask direct question"
    )
    response = dspy.OutputField(desc="Response to the user")
    follow_up_strategy = dspy.OutputField(
        desc="Internal note on next conversation strategy"
    )


class ElderChatModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.conversation_generator = dspy.ChainOfThought(ElderConversationSignature)

    def forward(self, context: List[dict], user_input: str, should_ask_question: bool):
        result = self.conversation_generator(
            context=context,
            user_input=user_input,
            should_ask_question=should_ask_question,
        )
        return result.response, result.follow_up_strategy


def prepare_examples():
    examples = [
        dspy.Example(
            context=[{"role": "user", "content": "I grew up in Ohio"}],
            user_input="I grew up in Ohio",
            should_ask_question=False,
            response="Those Midwest summers must have left you with some wonderful memories.",
            follow_up_strategy="Used affirmative statement to encourage sharing about childhood memories",
        ),
        dspy.Example(
            context=[
                {"role": "user", "content": "Yes, we used to have big family picnics"}
            ],
            user_input="Yes, we used to have big family picnics",
            should_ask_question=True,
            response="What was your favorite dish at these family gatherings?",
            follow_up_strategy="Asked specific but open-ended question about a detail mentioned",
        ),
        dspy.Example(
            context=[
                {"role": "user", "content": "My grandmother made the best apple pie"}
            ],
            user_input="My grandmother made the best apple pie",
            should_ask_question=False,
            response="Grandmothers have such a special way of making everything taste like love.",
            follow_up_strategy="Used emotional reflection to deepen the conversation",
        ),
        dspy.Example(
            context=[{"role": "user", "content": "I worked as a teacher for 35 years"}],
            user_input="I worked as a teacher for 35 years",
            should_ask_question=True,
            response="What grade level did you enjoy teaching the most?",
            follow_up_strategy="Focused question on positive aspects of their career",
        ),
    ]
    return examples


class ElderChatbot:
    def __init__(self):
        self.max_messages = 15
        self.messages = deque(maxlen=self.max_messages)
        self.chat_module = ElderChatModule()
        self.should_ask_question = True
        self.user_name = None
        self.user_image = None

        examples = prepare_examples()
        optimizer = BootstrapFewShot(max_labeled_demos=4)
        self.chat_module = optimizer.compile(self.chat_module, trainset=examples)

        self.messages.append(
            {
                "role": "system",
                "content": """You are a friendly and patient chatbot speaking with an elderly person.
            Engage naturally, alternating between gentle questions and affirming statements.""",
            }
        )

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_bot_response(self, user_input: str) -> str:
        self.add_message("user", user_input)

        response, strategy = self.chat_module(
            context=list(self.messages),
            user_input=user_input,
            should_ask_question=self.should_ask_question,
        )

        # Toggle between questions and affirmative statements
        self.should_ask_question = not self.should_ask_question

        self.add_message("assistant", response)
        return response

    def generate_story(self):
        from groq.types.chat import (
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
        )

        story_prompt = ChatCompletionSystemMessageParam(
            role="system",
            content="""Create a blog post from the following conversation. 
            The story should:
            1. Have an engaging title
            2. Be structured in clear sections
            3. Focus on the most interesting life stories and insights shared
            4. Include direct quotes when relevant
            5. Have a thoughtful conclusion
            6. Be between 500-1000 words
            7. Don't use markdown, because the final format will be a pdf.
            Format the response with the title on top, followed by the content in paragraphs.""",
        )

        user_message = ChatCompletionUserMessageParam(
            role="user", content=str(list(self.messages))
        )

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[story_prompt, user_message], model="llama-3.1-70b-versatile"
        )
        blog_post = chat_completion.choices[0].message.content
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
            user_input = input("You (type 'quit' to end): ")
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
