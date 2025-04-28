# app.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Groq LLM Setup
llm = ChatGroq(
    api_key="YOUR_KEY",  # <-- Replace with your actual Groq API Key
    model="llama3-8b-8192"     # You can also use llama3-70b-8192
)

# 2. Prompt Template
intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an intent classification system.
Analyze the following user message and output only the INTENT name, nothing else.

Possible INTENTS:
- BookFlight
- WeatherQuery
- PlayMusic
- TellJoke
- SetReminder

User Message: {user_input}

INTENT:
"""
)

# 3. Create Intent Detection Chain
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

# 4. Mock Task Functions
def perform_task(intent, user_input):
    if intent == "BookFlight":
        return f"📅 Booking flight as per details: '{user_input}'!"
    elif intent == "WeatherQuery":
        return "☁️ Fetching latest weather updates..."
    elif intent == "PlayMusic":
        return "🎵 Playing your favorite music..."
    elif intent == "TellJoke":
        return "😂 Here's a joke: Why don't scientists trust atoms? Because they make up everything!"
    elif intent == "SetReminder":
        return "⏰ Setting a reminder for you!"
    else:
        return "🤔 Sorry, I couldn't recognize the task."

# 5. Streamlit UI
st.title("🚀 Groq Intent Detection App")
st.write("Enter your command, and I'll detect the intent and perform the task!")

user_input = st.text_input("Your Message:")

if st.button("Detect & Perform"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Detecting intent..."):
            detected_intent = intent_chain.run(user_input).strip()

        st.success(f"✅ Detected Intent: `{detected_intent}`")

        # Perform based on Intent
        task_output = perform_task(detected_intent, user_input)
        st.info(task_output)
