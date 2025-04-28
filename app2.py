# app.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

# 1. Groq LLM Setup
llm = ChatGroq(
    api_key="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",  # Replace with your Groq API Key
    model="llama3-8b-8192"         # Updated active model
)

# 2. Prompt Template for Intent Detection
intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an intent classification system.
Analyze the following user message and output only the INTENT name, nothing else.

Possible INTENTS:
- AddNumbers
- SubtractNumbers
- MultiplyNumbers
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

# 4. Extract Numbers from User Input
def extract_numbers(text):
    return list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", text)))

# 5. Perform task based on intent
def perform_task(intent, user_input):
    numbers = extract_numbers(user_input)

    if intent == "AddNumbers":
        if len(numbers) >= 2:
            result = sum(numbers)
            return f"➕ The sum of the numbers is {result}."
        else:
            return "❗ Please provide at least two numbers to add."
    
    elif intent == "SubtractNumbers":
        if len(numbers) >= 2:
            result = numbers[0] - numbers[1]
            return f"➖ The result of subtraction is {result}."
        else:
            return "❗ Please provide two numbers to subtract."
    
    elif intent == "MultiplyNumbers":
        if len(numbers) >= 2:
            result = numbers[0] * numbers[1]
            return f"✖️ The product of the numbers is {result}."
        else:
            return "❗ Please provide two numbers to multiply."

    elif intent == "BookFlight":
        return f"📅 Booking flight with details: '{user_input}'!"

    elif intent == "WeatherQuery":
        return "☁️ Fetching the latest weather report..."

    elif intent == "PlayMusic":
        return "🎵 Playing some music for you..."

    elif intent == "TellJoke":
        return "😂 Why don't skeletons fight each other? They don't have the guts."

    elif intent == "SetReminder":
        return "⏰ Setting a reminder for you!"

    else:
        return "🤔 Sorry, I couldn't understand what task to perform."

# 6. Streamlit UI
st.title("🛠️ Groq LLM - Intent Detection & Execution App")
st.write("Enter your command below, and I'll detect the intent and perform the corresponding task!")

user_input = st.text_input("Your Message:")

if st.button("Detect Intent & Perform Task"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some input.")
    else:
        with st.spinner("🔎 Detecting intent..."):
            detected_intent = intent_chain.run(user_input).strip()

        st.success(f"✅ Detected Intent: `{detected_intent}`")

        task_output = perform_task(detected_intent, user_input)
        st.info(task_output)
