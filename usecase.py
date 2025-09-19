import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import httpx
from typing import List, Dict, Optional
import os

tiktoken_cache_dir = r"C:\Users\GenAICHNSIRUSR111\Eligibility Advisor\tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))
CUSTOMER_DATA = [
    {
        "id": "CUST001",
        "name": "John Smith",
        "email": "john.smith@email.com",
        "phone": "+1-555-0101",
        "account_status": "Active",
        "credit_score": 750,
        "current_credit_limit": 5000,
        "eligible_credit_limit": 8000,
        "monthly_income": 6000,
        "employment_status": "Employed",
        "account_age_months": 24,
        "is_eligible": True,
        "risk_level": "Low"
    },
    {
        "id": "CUST002", 
        "name": "Sarah Johnson",
        "email": "sarah.j@email.com",
        "phone": "+1-555-0102",
        "account_status": "Active",
        "credit_score": 680,
        "current_credit_limit": 3000,
        "eligible_credit_limit": 4500,
        "monthly_income": 4500,
        "employment_status": "Employed",
        "account_age_months": 18,
        "is_eligible": True,
        "risk_level": "Medium"
    },
    {
        "id": "CUST003",
        "name": "Mike Davis",
        "email": "mike.davis@email.com", 
        "phone": "+1-555-0103",
        "account_status": "Inactive",
        "credit_score": 620,
        "current_credit_limit": 2000,
        "eligible_credit_limit": 0,
        "monthly_income": 3500,
        "employment_status": "Unemployed",
        "account_age_months": 12,
        "is_eligible": False,
        "risk_level": "High"
    },
    {
        "id": "CUST004",
        "name": "Emily Wilson",
        "email": "emily.w@email.com",
        "phone": "+1-555-0104", 
        "account_status": "Active",
        "credit_score": 700,
        "current_credit_limit": 4000,
        "eligible_credit_limit": 6000,
        "monthly_income": 5200,
        "employment_status": "Employed",
        "account_age_months": 30,
        "is_eligible": True,
        "risk_level": "Low"
    },
    {
        "id": "CUST005",
        "name": "David Brown",
        "email": "david.brown@email.com",
        "phone": "+1-555-0105",
        "account_status": "Active", 
        "credit_score": 590,
        "current_credit_limit": 1500,
        "eligible_credit_limit": 2000,
        "monthly_income": 3200,
        "employment_status": "Part-time",
        "account_age_months": 8,
        "is_eligible": True,
        "risk_level": "High"
    },
    {
        "id": "CUST006",
        "name": "Lisa Garcia",
        "email": "lisa.garcia@email.com",
        "phone": "+1-555-0106",
        "account_status": "Suspended",
        "credit_score": 540,
        "current_credit_limit": 1000,
        "eligible_credit_limit": 0,
        "monthly_income": 2800,
        "employment_status": "Employed",
        "account_age_months": 6,
        "is_eligible": False,
        "risk_level": "High"
    },
    {
        "id": "CUST007",
        "name": "Robert Miller",
        "email": "robert.m@email.com", 
        "phone": "+1-555-0107",
        "account_status": "Active",
        "credit_score": 780,
        "current_credit_limit": 7000,
        "eligible_credit_limit": 12000,
        "monthly_income": 8500,
        "employment_status": "Employed",
        "account_age_months": 36,
        "is_eligible": True,
        "risk_level": "Low"
    },
    {
        "id": "CUST008",
        "name": "Amanda Taylor",
        "email": "amanda.t@email.com",
        "phone": "+1-555-0108",
        "account_status": "Active",
        "credit_score": 650,
        "current_credit_limit": 2500,
        "eligible_credit_limit": 3500,
        "monthly_income": 4000,
        "employment_status": "Employed", 
        "account_age_months": 15,
        "is_eligible": True,
        "risk_level": "Medium"
    },
    {
        "id": "CUST009",
        "name": "Chris Anderson",
        "email": "chris.a@email.com",
        "phone": "+1-555-0109", 
        "account_status": "Inactive",
        "credit_score": 600,
        "current_credit_limit": 1800,
        "eligible_credit_limit": 0,
        "monthly_income": 3800,
        "employment_status": "Self-employed",
        "account_age_months": 10,
        "is_eligible": False,
        "risk_level": "High"
    },
    {
        "id": "CUST010",
        "name": "Jennifer Lee",
        "email": "jennifer.lee@email.com",
        "phone": "+1-555-0110",
        "account_status": "Active",
        "credit_score": 710,
        "current_credit_limit": 3500,
        "eligible_credit_limit": 5500,
        "monthly_income": 5800,
        "employment_status": "Employed",
        "account_age_months": 22,
        "is_eligible": True,
        "risk_level": "Low"
    }
]

class CustomerVectorStore:
    """Manages customer data in vector database for semantic search"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.customer_lookup = {}
        self._initialize_vector_store()
    
    def _create_searchable_text(self, customer: Dict) -> str:
        """Create searchable text representation of customer data"""
        return f"""
        Customer ID: {customer['id']}
        Name: {customer['name']}
        Email: {customer['email']}
        Phone: {customer['phone']}
        Account Status: {customer['account_status']}
        Employment: {customer['employment_status']}
        Credit Score: {customer['credit_score']}
        Income: ${customer['monthly_income']:,}
        Risk Level: {customer['risk_level']}
        Eligibility: {'Eligible' if customer['is_eligible'] else 'Not Eligible'}
        """
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store with customer data"""
        try:
            documents = []
            
            for customer in CUSTOMER_DATA:
                searchable_text = self._create_searchable_text(customer)
                
                doc = Document(
                    page_content=searchable_text,
                    metadata={
                        "customer_id": customer['id'],
                        "name": customer['name'],
                        "email": customer['email'],
                        "phone": customer['phone']
                    }
                )
                documents.append(doc)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=50
                )
                chunked_documents = text_splitter.split_documents(documents)
                self.customer_lookup[customer['id']] = customer
            self.vector_store = FAISS.from_documents(chunked_documents, self.embeddings)
            
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
    
    def search_customers(self, query: str, k: int = 3) -> List[Dict]:
        """Search for customers using semantic similarity"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                customer_id = doc.metadata.get('customer_id')
                if customer_id in self.customer_lookup:
                    results.append(self.customer_lookup[customer_id])
            
            return results
            
        except Exception as e:
            st.error(f"Error searching customers: {str(e)}")
            return []
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict]:
        """Get customer by exact ID match"""
        return self.customer_lookup.get(customer_id)
    
    def search_by_contact_info(self, query: str) -> Optional[Dict]:
        """Search by exact email or phone match"""
        query_lower = query.lower()
        
        for customer in CUSTOMER_DATA:
            if (query_lower == customer['email'].lower() or 
                query_lower == customer['phone'].lower() or
                query_lower == customer['name'].lower()):
                return customer
        
        return None

def format_customer_info(customer: Dict) -> str:
    """Format customer information for display"""
    eligibility_emoji = "✅" if customer['is_eligible'] else "❌"
    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(customer['risk_level'], "⚪")
    
    return f"""
## 📋 Customer Information

**Customer ID:** {customer['id']}  
**Name:** {customer['name']}  
**Email:** {customer['email']}  
**Phone:** {customer['phone']}  

### 💳 Account Details
- **Status:** {customer['account_status']}
- **Account Age:** {customer['account_age_months']} months
- **Credit Score:** {customer['credit_score']}
- **Risk Level:** {risk_emoji} {customer['risk_level']}

### 💰 Financial Information
- **Monthly Income:** ${customer['monthly_income']:,}
- **Employment:** {customer['employment_status']}
- **Current Credit Limit:** ${customer['current_credit_limit']:,}
- **Eligible Credit Limit:** ${customer['eligible_credit_limit']:,}

### 🎯 Eligibility Status
{eligibility_emoji} **{'Eligible' if customer['is_eligible'] else 'Not Eligible'}** for credit limit increase

{get_recommendations(customer)}
"""

def get_recommendations(customer: Dict) -> str:
    """Generate personalized recommendations based on customer profile"""
    recommendations = []
    
    if customer['credit_score'] < 650:
        recommendations.append("💡 **Improve Credit Score:** Consider paying down existing balances and ensuring all payments are made on time.")
    
    if customer['account_status'] != 'Active':
        recommendations.append("⚠️ **Account Status:** Reactivate your account to become eligible for credit increases.")
    
    if customer['employment_status'] == 'Unemployed':
        recommendations.append("💼 **Employment:** Secure stable employment to improve eligibility.")
    
    if customer['monthly_income'] < 4000:
        recommendations.append("📈 **Income:** Increasing monthly income could improve credit limit eligibility.")
    
    if customer['account_age_months'] < 12:
        recommendations.append("⏳ **Account History:** Build a longer account history for better eligibility.")
    
    if not recommendations and customer['is_eligible']:
        recommendations.append("🎉 **Great Profile:** You have an excellent credit profile! Consider applying for the credit increase.")
    
    return "### 📝 Recommendations\n" + "\n".join(recommendations) if recommendations else ""

st.set_page_config(
    page_title="LLOYDS Credit Limit Eligibility Advisor",
    page_icon="💳",
    layout="wide"
)

client = httpx.Client(
    verify=False,
    timeout=60.0,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
)

def initialize_components():
    """Initialize LLM, embeddings, and vector store"""
    try:
        embeddings = OpenAIEmbeddings(
            base_url = "https://genailab.tcs.in/",
            model ='azure/genailab-maas-text-embedding-3-large',
            api_key = "sk-tDWgRqKO2NjPs0jx2Ls3lQ",
            http_client = client
        )
        
        llm = ChatOpenAI(
            base_url="https://genailab.tcs.in/",
            model="azure/genailab-maas-gpt-4o",
            api_key="sk-JmVEeaH6p90azCuJwyliJQ",
            http_client=client,
            temperature=0.0,
            max_tokens=1500,
            request_timeout=60,
            max_retries=3
        )
        
        vector_store = CustomerVectorStore(embeddings)
        
        return llm, vector_store, True
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, False

def main():
    st.title("💳 Credit Limit Increase Eligibility Advisor")
    
    if "components_initialized" not in st.session_state:
        with st.spinner("Initializing AI components..."):
            llm, vector_store, success = initialize_components()
            
            if success:
                st.session_state.llm = llm
                st.session_state.vector_store = vector_store
                st.session_state.components_initialized = True
                st.success("AI components initialized successfully!")
            else:
                st.error("Failed to initialize components. Please refresh the page.")
                return
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Customer Search")
        
        search_query = st.text_input(
            "Enter customer details:",
            placeholder="Name, Email, Phone, or Customer ID",
            help="Search by any customer information"
        )
        
        if st.button("🔍 Search Customer", use_container_width=True):
            if search_query:
                with st.spinner("Searching customers..."):
                    vector_store = st.session_state.vector_store
                    
                    exact_match = vector_store.search_by_contact_info(search_query)
                    if exact_match:
                        st.session_state.current_customer = exact_match
                        st.session_state.search_results = [exact_match]
                        st.success(f"Found customer: {exact_match['name']}")
                    else:
                        results = vector_store.search_customers(search_query, k=3)
                        if results:
                            st.session_state.search_results = results
                            st.session_state.current_customer = results[0]
                            st.success(f"Found {len(results)} matching customers")
                        else:
                            st.error("No customers found matching your search")
        
        if "search_results" in st.session_state and st.session_state.search_results:
            st.subheader("Search Results")
            for i, customer in enumerate(st.session_state.search_results):
                if st.button(
                    f"{customer['name']} ({customer['id']})",
                    key=f"customer_{i}",
                    use_container_width=True
                ):
                    st.session_state.current_customer = customer
        
        st.divider()
        
        # Show sample searches
        st.markdown("**Try searching for:**")
        st.code("john.smith@email.com")
        st.code("Sarah Johnson") 
        st.code("CUST003")
        st.code("+1-555-0104")
        
        st.divider()
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if "current_customer" in st.session_state:
                del st.session_state.current_customer
            st.rerun()
    
    if "advisor_agent" not in st.session_state:
        try:
            memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )

            agent_prompt = PromptTemplate(
                input_variables=["history", "input"],
                template="""You are a Credit Limit Eligibility Advisor AI assistant. You help customers check their account status, credit eligibility, and provide financial guidance.

Guidelines for responses:
1. Be professional, helpful, and friendly
2. Always verify customer identity by id or email or phone number before providing sensitive information
3. Provide clear eligibility status and explanations
4. Include relevant financial metrics when discussing eligibility
5. Offer actionable advice for improving credit eligibility when applicable
6. Use emojis appropriately to make responses engaging
7. If customer not found, suggest using the search function
8. Format monetary amounts with proper currency symbols and commas
9. Explain the factors affecting credit decisions

When customer information is available, provide comprehensive analysis including:
- Account status verification
- Current vs eligible credit limits
- Credit Limit Eligible Status in single line example: (Yes/No)
- Credit score and risk assessment
- Employment and income verification  
- Personalized recommendations for credit improvement

Always maintain confidentiality and only discuss information for the specific customer being inquired about.

Previous conversation:
{history}

Current query: {input}

Response:"""
            )

            st.session_state.advisor_agent = ConversationChain(
                llm=st.session_state.llm,
                memory=memory,
                prompt=agent_prompt,
                verbose=False
            )

        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            return

    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = """Welcome to the Credit Limit Advisor! 🏦

I'm here to help you with:
- Account status verification
- Credit eligibility assessment  
- Credit limit information
- Risk analysis and recommendations

**Use the search panel on the left to find customer information, or ask me any questions about credit eligibility!**"""

        st.session_state.messages.append(
            {"role": "assistant", "content": welcome_msg}
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about credit eligibility..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    context_prompt = prompt
                    if "current_customer" in st.session_state:
                        customer_context = format_customer_info(st.session_state.current_customer)
                        context_prompt = f"Current Customer Context:\n{customer_context}\n\nUser Query: {prompt}"

                    response = st.session_state.advisor_agent.predict(input=context_prompt)
                    st.markdown(response)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )

if __name__ == "__main__":
    main()
