import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import os
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="AI Travel Guide Agent",
    page_icon="✈️",
    layout="wide"
)


def create_travel_tools():
    """Create specialized tools for travel planning"""

    def get_travel_suggestions(query):
        """Tool for getting travel suggestions based on preferences"""
        suggestions = f"""
        Based on your query about {query}, here are some travel suggestions:
        
        🏖️ **Beach Destinations**: Consider coastal areas with warm weather
        🏔️ **Mountain Adventures**: Look into hiking trails and scenic viewpoints  
        🏛️ **Cultural Sites**: Museums, historical landmarks, and local traditions
        🍽️ **Food Experiences**: Local cuisines and must-try restaurants
        🎯 **Activities**: Adventure sports, tours, and unique experiences
        
        Would you like me to elaborate on any specific aspect?
        """
        return suggestions

    def budget_calculator(details):
        """Tool for calculating travel budgets"""
        budget_info = f"""
        💰 **Budget Estimation for {details}:**
        
        **Accommodation**: $50-200 per night depending on type
        **Meals**: $30-80 per day (budget to luxury)
        **Transportation**: Varies by distance and mode
        **Activities**: $20-100 per activity
        **Miscellaneous**: 10-20% of total budget
        
        💡 **Money-saving tips:**
        - Book flights in advance
        - Consider off-season travel
        - Look for package deals
        - Use local transportation
        """
        return budget_info

    def weather_advisor(location_time):
        """Tool for weather and best time to visit advice"""
        weather_info = f"""
        🌤️ **Weather & Best Time to Visit {location_time}:**
        
        **Research the seasonal patterns for your destination:**
        - Peak season: Higher prices but best weather
        - Shoulder season: Good balance of weather and prices
        - Off season: Budget-friendly but check weather conditions
        
        **What to check:**
        - Temperature ranges
        - Rainfall patterns
        - Local festivals and events
        - Crowd levels
        """
        return weather_info

    tools = [
        Tool(
            name="Travel Suggestions",
            func=get_travel_suggestions,
            description="Get travel destination and activity suggestions"
        ),
        Tool(
            name="Budget Calculator",
            func=budget_calculator,
            description="Calculate travel budgets and get money-saving tips"
        ),
        Tool(
            name="Weather Advisor",
            func=weather_advisor,
            description="Get weather information and best time to visit advice"
        )
    ]

    return tools


def initialize_travel_agent():
    """Initialize the travel guide agent"""

    # Initialize the Mistral model
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.7,
        max_tokens=800
    )

    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

    # Define specialized travel guide prompt
    travel_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""You are an expert AI Travel Guide Agent with extensive knowledge about destinations worldwide. 
Your role is to help users plan amazing trips by providing personalized recommendations, practical advice, and insider tips.

**Your Expertise Includes:**
🌍 Destination recommendations based on interests, budget, and season
🏨 Accommodation suggestions (hotels, hostels, vacation rentals)
✈️ Transportation options and booking tips
🗓️ Itinerary planning and time management
💰 Budget planning and money-saving strategies
🍽️ Local cuisine and restaurant recommendations
🎯 Activities, attractions, and hidden gems
📋 Travel documentation and visa requirements
🧳 Packing lists and travel gear suggestions
🚨 Safety tips and cultural etiquette

**Guidelines:**
- Always ask clarifying questions to provide personalized advice
- Consider budget, travel dates, interests, and travel style
- Provide practical, actionable recommendations
- Include both popular attractions and off-the-beaten-path suggestions
- Mention seasonal considerations and weather
- Suggest sustainable and responsible travel practices

Previous conversation:
{history}

Current query: {input}

Travel Guide Response:"""
    )

    # Create the conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=travel_prompt,
        verbose=False
    )

    return conversation


def main():
    # Header
    st.title("✈️ AI Travel Guide Agent")
    st.markdown("### Your Personal Travel Planning Assistant")

    # Create columns for better layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 🎒 Quick Travel Tools")

        # Quick travel planning tools
        with st.expander("📍 Destination Finder"):
            travel_style = st.selectbox(
                "Travel Style:",
                ["Adventure", "Relaxation", "Culture", "Food", "Nature", "History"]
            )
            budget_range = st.selectbox(
                "Budget Range:",
                ["Budget ($0-50/day)", "Mid-range ($50-150/day)",
                 "Luxury ($150+/day)"]
            )
            season = st.selectbox(
                "Preferred Season:",
                ["Spring", "Summer", "Fall", "Winter", "Flexible"]
            )

            if st.button("Get Destination Ideas"):
                suggestion = f"Looking for {travel_style.lower()} destinations in {season.lower()} with a {budget_range.lower()} budget"
                st.session_state.quick_query = suggestion

        with st.expander("🗓️ Trip Duration Helper"):
            trip_length = st.slider("Trip Length (days):", 1, 30, 7)
            destination_type = st.radio(
                "Destination Type:",
                ["Single City", "Multi-City", "Road Trip", "Island Hopping"]
            )

            if st.button("Plan My Trip"):
                planning_query = f"Help me plan a {trip_length}-day {destination_type.lower()} trip"
                st.session_state.quick_query = planning_query

        with st.expander("💰 Budget Planner"):
            travelers = st.number_input("Number of Travelers:", 1, 10, 1)
            trip_days = st.number_input("Trip Duration (days):", 1, 30, 7)

            if st.button("Calculate Budget"):
                budget_query = f"Help me budget for {travelers} travelers on a {trip_days}-day trip"
                st.session_state.quick_query = budget_query

    with col1:
        # Sidebar for configuration
        with st.sidebar:
            st.title("⚙️ Configuration")

            # API Key input
            api_key = st.text_input(
                "Enter your Mistral API Key:",
                type="password",
                help="Get your API key from https://console.mistral.ai/"
            )

            # Model selection
            model_options = [
                "mistral-small-latest",
                "mistral-medium-latest",
                "mistral-large-latest"
            ]
            selected_model = st.selectbox("Select Model:", model_options)

            # Temperature slider
            temperature = st.slider("Creativity Level:", 0.0, 1.0, 0.7, 0.1)

            # Max tokens slider
            max_tokens = st.slider("Response Length:", 200, 1500, 800, 100)

            st.markdown("---")
            st.markdown("### 🚀 Quick Actions")

            # Preset queries
            if st.button("🌟 Popular Destinations"):
                st.session_state.quick_query = "What are the most popular travel destinations right now?"

            if st.button("🎒 Packing Tips"):
                st.session_state.quick_query = "Give me essential packing tips for international travel"

            if st.button("📱 Travel Apps"):
                st.session_state.quick_query = "What are the best travel apps I should have?"

            if st.button("🛡️ Safety Tips"):
                st.session_state.quick_query = "What are important travel safety tips?"

            # Clear conversation button
            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

        # Check if API key is provided
        if not api_key:
            st.warning(
                "⚠️ Please enter your Mistral API key in the sidebar to start planning your trip!")
            st.info("💡 **Sample Questions to Ask:**")
            st.markdown("""
            - "I want to plan a 10-day trip to Europe in summer for $3000"
            - "What are the best destinations for solo female travelers?"
            - "Help me create an itinerary for Japan in spring"
            - "I need budget-friendly options for Southeast Asia"
            - "What should I pack for a winter trip to Iceland?"
            """)
            return

        # Set the API key in environment variable
        os.environ["MISTRAL_API_KEY"] = api_key

        # Initialize the travel agent if not already done
        if "travel_agent" not in st.session_state:
            try:
                # Update model parameters
                llm = ChatMistralAI(
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                memory = ConversationBufferMemory(
                    memory_key="history",
                    return_messages=True
                )

                travel_prompt = PromptTemplate(
                    input_variables=["history", "input"],
                    template="""You are an expert AI Travel Guide Agent with extensive knowledge about destinations worldwide. 
Your role is to help users plan amazing trips by providing personalized recommendations, practical advice, and insider tips.

**Your Expertise Includes:**
🌍 Destination recommendations based on interests, budget, and season
🏨 Accommodation suggestions (hotels, hostels, vacation rentals)
✈️ Transportation options and booking tips
🗓️ Itinerary planning and time management
💰 Budget planning and money-saving strategies
🍽️ Local cuisine and restaurant recommendations
🎯 Activities, attractions, and hidden gems
📋 Travel documentation and visa requirements
🧳 Packing lists and travel gear suggestions
🚨 Safety tips and cultural etiquette

**Guidelines:**
- Always ask clarifying questions to provide personalized advice
- Consider budget, travel dates, interests, and travel style
- Provide practical, actionable recommendations
- Include both popular attractions and off-the-beaten-path suggestions
- Mention seasonal considerations and weather
- Suggest sustainable and responsible travel practices

Previous conversation:
{history}

Current query: {input}

Travel Guide Response:"""
                )

                st.session_state.travel_agent = ConversationChain(
                    llm=llm,
                    memory=memory,
                    prompt=travel_prompt,
                    verbose=False
                )

            except Exception as e:
                st.error(f"❌ Error initializing travel agent: {str(e)}")
                return

        # Initialize chat messages in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message
            welcome_msg = """👋 **Welcome to your AI Travel Guide!** 

I'm here to help you plan the perfect trip! I can assist you with:

🗺️ **Destination recommendations** based on your interests
💰 **Budget planning** and money-saving tips  
🗓️ **Itinerary creation** for any trip length
🏨 **Accommodation suggestions** for all budgets
🍽️ **Local food and restaurant recommendations**
🎯 **Activities and hidden gems** at your destination
📋 **Travel documentation** and visa information
🧳 **Packing lists** tailored to your trip

**To get started, tell me:**
- Where are you thinking of going?
- What's your budget range?
- How long is your trip?
- What interests you most?

Let's plan an amazing adventure together! ✈️"""

            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg})

        # Handle quick query from sidebar or tools
        if "quick_query" in st.session_state and st.session_state.quick_query:
            query = st.session_state.quick_query
            st.session_state.messages.append(
                {"role": "user", "content": query})

            # Get response
            with st.spinner("🔍 Planning your trip..."):
                try:
                    response = st.session_state.travel_agent.predict(
                        input=query)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg})

            # Clear the quick query
            del st.session_state.quick_query
            st.rerun()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about travel planning..."):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get travel agent response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Planning your trip..."):
                    try:
                        response = st.session_state.travel_agent.predict(
                            input=prompt)
                        st.markdown(response)

                        # Add assistant response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"❌ Error: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
