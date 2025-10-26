import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import numpy as np
import os
import time
import sqlite3
from datetime import datetime
import re 

# --- 1. Model Training and Setup ---

# Define the expected path for the dataset
# NOTE: Replace 'Phishing_Email.csv' with the actual file name if it's different.
DATA_PATH = 'Phishing_Email.csv' 
# Define the SQLite database file path
DB_FILE = 'analysis_history.db'

# Sample emails for the 'Analyze Sample' feature
SAMPLE_EMAILS = {
    # ----------------------------------------------------------------------
    # LOW RISK (SAFE) - Expected to be "Legitimate Email" -> SAFE (Low Risk)
    # ----------------------------------------------------------------------
    "1. Legitimate: Team Meeting Reminder (Low Risk)": {
        "text": "Friendly reminder that our quarterly product review meeting is scheduled for tomorrow at 10 AM. Please ensure you have finalized your slides. Find the updated agenda attached.",
        "sender": "jane.doe@company.com",
        "subject": "Q4 Product Review Meeting Agenda"
    },
    "2. Legitimate: Internal HR Memo (Low Risk)": {
        "text": "This is a non-urgent notification regarding the new vacation policy effective January 1st. Details are available on the internal intranet portal. No immediate action is required from employees at this time.",
        "sender": "hr-department@company.com",
        "subject": "FYI: Updated Vacation Policy for 2026"
    },
    
    # ----------------------------------------------------------------------
    # HIGH RISK (PHISHING) - REINFORCED FOR RELIABLE PHISHING OUTPUT
    # ----------------------------------------------------------------------
    "3. High Risk: Generic Account Update": {
        "text": "**URGENT ACTION REQUIRED!** Due to a recent system update, we require all users to reconfirm their account information by clicking 'Confirm Details' below. Your services may experience temporary interruption if this is not completed within 48 hours. Click here: http://account-update-portal.info",
        "sender": "system_update@info.com",
        "subject": "Important: Account Reconfirmation Required"
    },
    # ----------------------------------------------------------------------
    # CRITICAL EDIT: Enhanced Sample 4 for reliable Phishing classification
    "4. High Risk: URGENT Billing Issue": {
        "text": "**IMMEDIATE ATTENTION REQUIRED!** We have detected a critical error with your billing profile. Your account will be **suspended** within 24 hours if you do not immediately **verify** your details. Click the link below to resolve the **unusual charge** of $5.99 now: http://billing-portal-verify.net/fixbilling",
        "sender": "billing@service-records.net",
        "subject": "URGENT: Billing Issue and Account Suspension Warning"
    },
    "5. High Risk: Urgent Account Lock": {
        "text": "**MANDATORY SECURITY VERIFICATION:** We have detected unusual activity on your account. Click the link below immediately to verify your password and prevent permanent suspension of service. Failure to act will result in loss of access. http://verify-now.net",
        "sender": "noreply@security.com",
        "subject": "URGENT: Your account has been temporarily locked"
    },
    "6. High Risk: Fake Invoice Threat": {
        "text": "SECURITY ALERT: Your invoice #49282 for $984.00 is now overdue. To avoid account suspension, click the link below to verify your details and pay the balance urgently. Failure to act will result in service termination. https://secure-billing.net/invoice49282",
        "sender": "billing.service@support.com",
        "subject": "Action Required: Overdue Payment Notice"
    },
}

# Check if the file exists before attempting to read
try:
    if not os.path.exists(DATA_PATH):
        st.error(f"Error: Dataset file '{DATA_PATH}' not found. Please ensure 'Phishing_Email.csv' is in the same directory.")
        st.stop()
except Exception as e:
    st.error(f"Error during file system check: {e}")
    st.stop()


@st.cache_resource
def train_model(data_path):
    """Loads data, trains the model, and returns the model and vectorizer."""
    try:
        # Load Data
        data = pd.read_csv(data_path, encoding='latin-1')
        
        # Ensure correct column names
        if 'Email Text' not in data.columns or 'Email Type' not in data.columns:
            if data.shape[1] >= 2:
                # Assuming the first two columns are the type and text if headers are missing/misnamed
                data.columns = ['Email Type', 'Email Text'] + list(data.columns[2:])
            else:
                 raise ValueError("CSV must have at least two columns: 'Email Type' and 'Email Text'.")

        # Data Cleaning
        data.dropna(subset=['Email Text'], inplace=True)
        data.drop_duplicates(subset=['Email Text'], inplace=True)
        
        # --- NEW ROBUSTNESS FIXES ---
        # 1. Standardize text to lowercase (ensures consistency)
        data['Email Text'] = data['Email Text'].astype(str).str.lower()
        
        # 2. Basic removal of non-alphanumeric characters (reduces noise)
        data['Email Text'] = data['Email Text'].apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', '', x)
        )
        
        # Training Setup
        mess = data['Email Text']
        cat = data['Email Type']
        mess_train, _, cat_train, _ = train_test_split(
            mess, cat, test_size=0.2, random_state=42, stratify=cat
        )

        # Feature Engineering (Vectorization)
        cv = CountVectorizer(stop_words='english')
        features = cv.fit_transform(mess_train)

        # Training Model
        model = MultinomialNB()
        model.fit(features, cat_train)

        return model, cv

    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        st.stop()


model, cv = train_model(DATA_PATH)

# --- 2. Data Extraction and Persistence Functions ---

def extract_metadata(email_text):
    """
    Attempts to extract a subject, sender-like email, and link from the text.
    """
    if not email_text:
        return "unknown-sender@example.com", "(No Text Provided)", None
        
    # Simple regex to find an email address
    sender_match = re.search(r'(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)', email_text)
    sender = sender_match.group(0) if sender_match else "unknown-sender@example.com"
    
    # Use the first line as a simulated subject
    subject = email_text.split('\n')[0].strip()
    subject = subject if len(subject) < 100 else subject[:97] + "..."
    if not subject:
        subject = "(No Subject Line Detected)"
        
    # Heuristic to find a potential link (http or https)
    link_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
    link = link_match.group(0) if link_match else None
    
    return sender, subject, link

def column_exists(cursor, table, column):
    """Helper function to check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    return any(col[1] == column for col in cursor.fetchall())

def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Initial table creation with all expected columns
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sender TEXT,
            subject TEXT,
            prediction TEXT,
            confidence REAL,
            full_email TEXT,
            status TEXT DEFAULT 'Pending'
        )
    """)
    
    # --- Database Migration Checks ---
    if not column_exists(c, 'analysis_results', 'sender'):
        c.execute("ALTER TABLE analysis_results ADD COLUMN sender TEXT")
    
    if not column_exists(c, 'analysis_results', 'subject'):
        c.execute("ALTER TABLE analysis_results ADD COLUMN subject TEXT")
        
    if not column_exists(c, 'analysis_results', 'full_email'):
        c.execute("ALTER TABLE analysis_results ADD COLUMN full_email TEXT")

    if not column_exists(c, 'analysis_results', 'status'):
        c.execute("ALTER TABLE analysis_results ADD COLUMN status TEXT DEFAULT 'Pending'")


    conn.commit()
    conn.close()

def map_category(pred, conf):
    """
    Maps the raw prediction to a 2-level dashboard category (Phishing or Safe).
    """
    # 1. CRITICAL: Clean the prediction string to ensure accurate matching
    clean_pred = str(pred).strip().lower()
    
    # 2. HIGH RISK (Phishing - RED): Any prediction of 'Phishing Email'
    if 'phishing' in clean_pred:
        return 'Phishing' 
         
    # 3. LOW RISK (Safe - GREEN): Anything else (i.e., model predicted 'Legitimate Email').
    else:
        return 'Safe'


def save_analysis(email_text, prediction, confidence, sender, subject):
    """Saves the analysis result to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("""
        INSERT INTO analysis_results (timestamp, sender, subject, prediction, confidence, full_email, status)
        VALUES (?, ?, ?, ?, ?, ?, 'Pending')
    """, (timestamp, sender, subject, prediction, confidence, email_text))
    
    conn.commit()
    conn.close()
    
def update_analysis_status(record_id, new_status):
    """
    Updates the status of a specific analysis result in the SQLite database.
    This includes an immediate visual confirmation via st.toast and
    forces a page refresh via st.rerun() to show the updated status.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        UPDATE analysis_results
        SET status = ?
        WHERE id = ?
    """, (new_status, record_id))
    conn.commit()
    conn.close()
    
    # Immediate visual confirmation and page reload
    st.toast(f"Record #{record_id} status changed to: {new_status}!", icon="✅")
    st.rerun()


def load_analysis(limit=None):
    """Loads all analysis results from the SQLite database, optionally limited."""
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT id, timestamp, sender, subject, prediction, confidence, full_email, status FROM analysis_results ORDER BY id DESC"
    if limit:
        query += f" LIMIT {limit}"
        
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if 'full_email' in df.columns:
        df['full_email'] = df['full_email'].fillna('').astype(str)

    # Use the globally defined map_category function
    df['dashboard_category'] = df.apply(
        lambda row: map_category(row['prediction'], row['confidence']), axis=1
    )
    return df

# Initialize the database when the app starts
init_db()


# --- 3. Prediction and Feature Analysis Functions ---

def get_top_phishing_tokens(email_text, vectorizer, model, top_n=8):
    """Identifies the top tokens in the email that contribute to the 'Phishing' classification."""
    
    input_vector = vectorizer.transform([email_text])
    
    if input_vector.nnz == 0:
        return []

    # Find the index for the 'Phishing Email' class
    try:
        phishing_index = np.where([c for c in model.classes_ if 'Phishing' in c])[0][0]
    except IndexError:
        return [] 
        
    email_features = input_vector.toarray()[0]
    phishing_log_probs = model.feature_log_prob_[phishing_index]
    
    scores = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(len(email_features)):
        if email_features[i] > 0:
            # Score contribution from this word toward the Phishing class
            score = phishing_log_probs[i] * email_features[i] 
            scores[feature_names[i]] = score

    # Sort by the score (higher score means stronger contribution to Phishing)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    return [word for word, score in sorted_scores[:top_n]]


def get_top_legitimate_tokens(email_text, vectorizer, model, top_n=8):
    """Identifies the top tokens in the email that contribute to the 'Legitimate' classification."""
    
    input_vector = vectorizer.transform([email_text])
    
    if input_vector.nnz == 0:
        return []

    # Find the index for the 'Legitimate Email' class
    try:
        legitimate_index = np.where([c for c in model.classes_ if 'Legitimate' in c])[0][0]
    except IndexError:
        return [] 
        
    email_features = input_vector.toarray()[0]
    legitimate_log_probs = model.feature_log_prob_[legitimate_index]
    
    scores = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(len(email_features)):
        if email_features[i] > 0:
            # Score contribution from this word toward the Legitimate class
            score = legitimate_log_probs[i] * email_features[i] 
            scores[feature_names[i]] = score

    # Sort by the score (higher score means stronger contribution to Legitimate)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    return [word for word, score in sorted_scores[:top_n]]


def predict_email(email_text):
    """Predicts the category and confidence of the given email text."""
    input_message = cv.transform([email_text])
    result = model.predict(input_message)
    prediction_text = result[0]
    
    # Debug print line (if you still want it)
    print(f"DEBUG: Raw Prediction Text: '{prediction_text}'")  
    
    probabilities = model.predict_proba(input_message)[0]
    
    # Initialize confidence to prevent NameError
    confidence = 0.0
    
    if prediction_text in model.classes_:
        class_index = np.where(model.classes_ == prediction_text)[0][0]
        confidence = probabilities[class_index]

    return prediction_text, confidence

# --- 4. Streamlit UI/UX and Layout & Action Callbacks ---

st.set_page_config(
    page_title="SecureScan - Phishing Detector",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS for high visibility and clean, card-based design
st.markdown("""
<style>
    /* Main Title Styling */
    h1 {
        font-size: 2.5rem;
        color: #172A4E; /* Deep Blue Header */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* Card Container */
    .stCard {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 20px;
        background-color: white;
    }
    /* Section Headers */
    .stCard h2, .stCard h3 {
        color: #172A4E;
        border-bottom: 2px solid #EEE;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    /* Button Styling */
    div.stButton > button {
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 1rem;
        font-weight: bold;
        width: 100%;
        margin-top: 10px;
    }
    /* Specific Alert Tags */
    .alert-tag {
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        min-width: 60px;
    }
    .high { background-color: #F8D7DA; color: #721C24; border: 1px solid #F5C6CB; }
    /* Removed .medium styling */
    .safe-tag { background-color: #D4EDDA; color: #155724; border: 1px solid #C3E6CB; }

    /* Dashboard Specific Colors */
    .stat-text-safe { color: #28A745; font-weight: bold; }
    /* Removed .stat-text-suspicious styling */
    .stat-text-phishing { color: #DC3545; font-weight: bold; }
    
    /* Result Box Styling (from previous iteration, kept for Analyze tab) */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 2px solid;
    }
    .safe-analysis {
        background-color: #E8FFF1; /* Light Green */
        border-color: #28A745; /* Dark Green */
    }
    .phishing-analysis {
        background-color: #FEE7E7; /* Lighter Red Background */
        border-color: #A52A2A; /* Deeper Red Border */
        box-shadow: 0 0 20px rgba(220, 53, 69, 0.3);
    }
    /* Removed .suspicious-analysis styling */
</style>
""", unsafe_allow_html=True)

# --- 5. Session State Initialization & Callbacks ---

# Initialize session state variables for UI components
if 'input_text_area' not in st.session_state:
    st.session_state.input_text_area = ""
if 'input_subject' not in st.session_state:
    st.session_state.input_subject = ""
if 'input_sender' not in st.session_state:
    st.session_state.input_sender = ""
if 'current_alert_id' not in st.session_state:
    st.session_state.current_alert_id = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None # Initialize last_prediction


# Callback for Sample Selector
def update_input_from_sample():
    """Callback function to load sample data instantly into the text areas."""
    sample_choice = st.session_state.sample_selector
    if sample_choice != "-- Paste Your Own Email Below --":
        sample = SAMPLE_EMAILS[sample_choice]
        st.session_state.input_text_area = sample['text']
        st.session_state.input_subject = sample['subject']
        st.session_state.input_sender = sample['sender']
    else:
        st.session_state.input_text_area = ""
        st.session_state.input_subject = ""
        st.session_state.input_sender = ""

# --- Action Callbacks ---
def quarantine_alert():
    """Marks the currently displayed alert as Quarantined."""
    if st.session_state.current_alert_id is not None:
        update_analysis_status(st.session_state.current_alert_id, 'Quarantined')

def delete_alert():
    """Marks the currently displayed alert as Deleted."""
    if st.session_state.current_alert_id is not None:
        update_analysis_status(st.session_state.current_alert_id, 'Deleted') 

# Updated function name and status string for clarity ("Marked as Safe")
def mark_as_safe_alert():
    """Marks the currently displayed alert as Marked as Safe."""
    if st.session_state.current_alert_id is not None:
        update_analysis_status(st.session_state.current_alert_id, 'Marked as Safe')

# --- Dashboard Rendering Function ---
def render_dashboard(df):
    """Renders the custom dashboard layout."""
    
    st.markdown("## Email Phishing Detection and Prevention Dashboard")
    st.markdown("---")
    
    # Reset the current alert ID at the start of rendering
    st.session_state.current_alert_id = None 

    if df.empty:
        st.info("No analysis history yet. Run a scan on the 'Analyze Email' tab to populate the dashboard.")
        return

    # Calculate statistics based on the 'dashboard_category' (Safe, Phishing - Suspicious is now Phishing)
    # Reindex only for Safe and Phishing
    summary = df['dashboard_category'].value_counts().reindex(['Safe', 'Phishing'], fill_value=0)
    total_scans = summary.sum()
    
    # 1. Main Columns (Left: Summary & Details, Right: Alerts & Actions)
    col_left, col_right = st.columns([5, 4])

    # --- LEFT COLUMN: Threat Summary & Email Details ---
    with col_left:
        # 1A. Threat Summary Card (Gauge/Donut Chart)
        with st.container(border=True):
            st.markdown("### Threat Summary")
            
            # Display total scanned emails just below the title
            st.markdown(f"**Total Emails Scanned:** **`{total_scans}`**", unsafe_allow_html=True)
            st.markdown("---") # Separator below the total count
            
            # Data for the Donut Chart (Only Safe and Phishing)
            chart_data = pd.DataFrame({
                'category': ['Phishing', 'Safe'], # Order Phishing first in the dataframe for better visualization
                'count': [summary.get('Phishing', 0), summary.get('Safe', 0)]
            })

            # Define the explicit color scale: Phishing must be Red, Safe must be Green
            chart_domain = ['Phishing', 'Safe']
            chart_range = ['#DC3545', '#28A745'] # Red for Phishing, Green for Safe

            if total_scans > 0:
                # Streamlit Vega-Lite chart for the Donut visualization
                st.vega_lite_chart(chart_data, {
                    'mark': {'type': 'arc', 'innerRadius': 60, 'outerRadius': 120},
                    'encoding': {
                        'theta': {"field": "count", "type": "quantitative", "stack": "normalize"},
                        'color': {
                            "field": "category", 
                            "type": "nominal", 
                            "scale": {"domain": chart_domain, "range": chart_range} # Explicitly map Phishing->Red and Safe->Green
                        },
                        'order': {"field": "count", "sort": "descending"}
                    },
                    'width': 250,
                    'height': 250,
                    'view': {'stroke': None}
                }, use_container_width=True) 
            else:
                st.markdown("<div style='height: 250px; text-align: center; padding-top: 100px;'><small>No scans to display.</small></div>", unsafe_allow_html=True)


        # 1B. Email Details Card (Most Recent PENDING Phishing)
        with st.container(border=True):
            st.markdown("### Email Details")
            
            # Filter for the latest PENDING phishing email
            critical_df = df[
                (df['dashboard_category'] == 'Phishing') & 
                (df['status'] == 'Pending')
            ].head(1)
            
            action_disabled = True
            
            if not critical_df.empty: 
                latest_alert = critical_df.iloc[0]
                st.session_state.current_alert_id = latest_alert['id'] # Set the ID for the actions column
                action_disabled = False
                
                full_email_text = latest_alert['full_email'] if latest_alert['full_email'] else "(No Email Body Found in History)"

                st.markdown(f"#### ID #{latest_alert['id']}: {latest_alert['subject']}") 
                
                st.text(f"Sender: {latest_alert['sender']}")
                st.text(f"Status: {latest_alert['status']} | Category: {latest_alert['dashboard_category']}") 
                
                try:
                    timestamp_formatted = datetime.strptime(latest_alert['timestamp'], '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y, %I:%M %p')
                except ValueError:
                    timestamp_formatted = latest_alert['timestamp'] 

                st.text(f"Date: {timestamp_formatted}")
                
                st.markdown("---")
                
                _, _, link = extract_metadata(full_email_text) 

                display_body = full_email_text[:350].replace('\n', ' ') + "..."

                st.markdown(display_body)
                
                if link:
                    # Link is now functional with target='_blank'
                    st.markdown(f"Potential Link: <a href='{link}' target='_blank' style='color: #0077B6; font-weight: bold;'>{link}</a>", unsafe_allow_html=True)
                
            else:
                # If no pending alerts, check the most recent PROCESSED one to show context
                processed_df = df[df['status'] != 'Pending'].head(1)
                if not processed_df.empty:
                    latest_alert = processed_df.iloc[0]
                    st.info(f"The most recent threat (ID #{latest_alert['id']}) has already been *{latest_alert['status']}*.")
                    st.markdown(f"#### Processed Alert: {latest_alert['subject']}")
                    st.text(f"Sender: {latest_alert['sender']}")
                    st.text(f"Status: {latest_alert['status']} | Category: {latest_alert['dashboard_category']}")
                else:
                    st.info("No phishing emails detected yet.")

    # --- RIGHT COLUMN: Recent Alerts & Actions ---
    with col_right:
        # 2A. Recent Alerts Card
        with st.container(border=True):
            st.markdown("### Recent Alerts")
            
            # Show the 5 most recent overall scans
            recent_alerts = df.head(5) 
            
            if not recent_alerts.empty:
                for index, row in recent_alerts.iterrows():
                    # Determine alert level and tag text (Only High or Low)
                    category = row['dashboard_category']
                    status = row['status']
                    tag_class = ""
                    tag_text = ""
                    
                    if status != 'Pending':
                        tag_class = "safe-tag" # Use low-risk color for processed items
                        tag_text = status.replace(' ', '_') # Show the status
                    elif category == 'Phishing':
                        tag_class = "high"
                        tag_text = "High"
                    else: # Must be 'Safe'
                        tag_class = "safe-tag"
                        tag_text = "Low"
                        
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #DDD; padding: 10px 0;'>
                        <div style='flex-grow: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;'>
                            <strong title="{row['subject']}">{row['subject']}</strong><br>
                            <small>ID #{row['id']} from {row['sender']}</small>
                        </div>
                        <span class='alert-tag {tag_class}'>{tag_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent alerts to display.")
                

        # 2B. Actions Card
        with st.container(border=True):
            st.markdown("### Actions")
            
            # Check if there is an alert ID set to enable the buttons
            if st.session_state.current_alert_id is None:
                st.info("No *pending* high-risk alerts to act upon. Scan an email or review history.")
            else:
                st.markdown(f"*Action Target: Alert ID #{st.session_state.current_alert_id}*")
                
                col_act1, col_act2, col_act3 = st.columns(3)
                
                with col_act1:
                    # Bind button click to callback function
                    st.button('🔴 Quarantine', key="act_quarantine", on_click=quarantine_alert, disabled=action_disabled, help="Isolate the email (Mock Action)", use_container_width=True)
                with col_act2:
                    st.button('⚫ Delete', key="act_delete", on_click=delete_alert, disabled=action_disabled, help="Permanently delete (Mock Action)", use_container_width=True)
                with col_act3:
                    # Updated callback to the new function name: mark_as_safe_alert
                    st.button('🟢 Mark as Safe', key="act_safe", on_click=mark_as_safe_alert, disabled=action_disabled, help="Mark as safe for future training (Mock Action)", use_container_width=True)

# --- Application Layout ---
st.title("🛡 SecureScan: Real-Time Phishing Detector")
st.markdown("---")

# Placeholder for the dynamic status indicator
status_container = st.empty()

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Analyze Email", "📚 Learn & Breakdown", "📊 Analysis History & Dashboard", "⚙ System Info"])


with tab1:
    st.subheader("1. Enter Email Content")
    
    # --- Sample Selector ---
    sample_choice = st.selectbox(
        "Or, quick-analyze a pre-loaded sample:",
        options=["-- Paste Your Own Email Below --"] + list(SAMPLE_EMAILS.keys()),
        index=0,
        key="sample_selector",
        on_change=update_input_from_sample # Calls the function to update state when a sample is chosen
    )

    # --- Manual Input Fields ---
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        # These now rely entirely on their session state keys set by init or callback
        st.text_input(
            "*Subject Line (Optional):*",
            key="input_subject"
        )
    with col_meta2:
        st.text_input(
            "*Sender Address (Optional):*",
            key="input_sender"
        )

    # Input Area (updates session state via its key)
    input_mess = st.text_area(
        '*Paste the full email body here:*',
        height=200,
        placeholder="e.g., Your account has been suspended. Click the secure link now to verify.",
        key="input_text_area"
    )
    
    # Analyze Button
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        # Use the key content directly for the check
        analyze_button = st.button('🚀 RUN PHISHING SCAN')

    if analyze_button and st.session_state.input_text_area.strip() != "":
        
        # --- Real-Time Status Indicator ---
        with status_container.container():
            st.info("Scanning in progress... Please wait.")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.005) # Simulated delay
                progress_bar.progress(i + 1)
            progress_bar.empty()
        
        # Get metadata from the session state keys
        sender = st.session_state.input_sender if st.session_state.input_sender.strip() else "unknown-sender@example.com"
        subject = st.session_state.input_subject if st.session_state.input_subject.strip() else "(No Subject Provided)"
        input_text = st.session_state.input_text_area
        
        # --- Prediction ---
        prediction, confidence = predict_email(input_text)
        
        # Calculate dashboard_category using the global map_category function (only Phishing or Safe now)
        dashboard_category = map_category(prediction, confidence) 
        
        # --- SAVE TO DATABASE ---
        save_analysis(input_text, prediction, confidence, sender, subject)

        # --- Display Result ---
        with status_container.container():
            # Only Phishing Check is required now (this covers all cases where 'Phishing Email' was predicted)
            if dashboard_category == 'Phishing':
                # High-Visibility PHISHING Result
                st.markdown(f"""
                <div class='result-box phishing-analysis'>
                    <span style='background-color: #DC3545; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;'>⚠ CRITICAL THREAT LEVEL</span>
                    <div style='font-size: 3rem; line-height: 1; margin-bottom: 10px; color: #DC3545;'>🚨</div>
                    <div style='font-size: 1.8rem; font-weight: bold; color: #DC3545;'>PHISHING ATTEMPT DETECTED!</div>
                    <p>
                        *Confidence Score (Raw Model Output):* <span style='color: #DC3545; font-weight: bold;'>{confidence:.2f}</span>
                    </p>
                    <hr style='border-top: 1px solid #DC3545;'>
                    <p style='font-size: 1.1rem; color: #A52A2A; font-weight: bold;'>
                        *IMMEDIATE ACTION REQUIRED:* This is a high-risk email. DO NOT click any links, DO NOT reply.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.toast("🚨 Phishing detected! See the breakdown in the 'Learn' tab.", icon='🚨')
            
            else: # Safe (Legitimate Email prediction)
                # High-Visibility SAFE Result
                st.markdown(f"""
                <div class='result-box safe-analysis'>
                    <span style='background-color: #28A745; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;'>🟢 LOW THREAT LEVEL</span>
                    <div style='font-size: 3rem; line-height: 1; margin-bottom: 10px; color: #28A745;'>🛡</div>
                    <div style='font-size: 1.8rem; font-weight: bold; color: #28A745;'>LOW RISK - LEGITIMATE</div>
                    <p><strong>Confidence: {confidence:.2f}</strong> | *CAUTION:* Always verify sender.</p>
                </div>
                """, unsafe_allow_html=True)
                st.toast("✅ Email marked as low risk.", icon='✅')

        # Update last_prediction in session state
        st.session_state.last_prediction = {
            'text': input_text,
            'prediction': prediction,
            'confidence': confidence,
            'sender': sender,
            'subject': subject,
            'dashboard_category': dashboard_category
        }

    elif analyze_button and st.session_state.input_text_area.strip() == "":
        status_container.warning("Please paste email content to analyze.")
        
    st.markdown("""
        <div style='margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee;'>
        <small>Note: This system provides an AI-based risk assessment. Always use human judgment.</small>
        </div>
    """, unsafe_allow_html=True)


with tab2:
    st.subheader("Suspicion Breakdown & Education")
    
    if 'last_prediction' in st.session_state and st.session_state.last_prediction is not None:
        pred_data = st.session_state.last_prediction
        dashboard_category = pred_data.get('dashboard_category', 'N/A')

        st.markdown(f"#### Last Analyzed Result: *{pred_data['prediction']}* | Dashboard Category: *{dashboard_category}* (Confidence: {pred_data['confidence']:.2f})")
        
        # --- HIGH RISK Breakdown (Phishing Keywords) ---
        if dashboard_category == 'Phishing':
            
            st.error(f"🚨 The model identified *High-risk* indicators.")
            
            # Use the Phishing keyword finder
            top_tokens = get_top_phishing_tokens(pred_data['text'], cv, model, top_n=8)
            
            if top_tokens:
                st.markdown(f"##### 🔑 Top Suspicious Keywords Found (High Risk):")
                
                # Use Red color for tags
                tag_color = "#FDC7D7" 
                tag_text_color = "#DC3545" 
                
                token_html = "".join([f"<span style='background-color: {tag_color}; color: {tag_text_color}; padding: 5px 10px; margin: 4px; border-radius: 6px; display: inline-block; font-weight: bold;'>{token}</span>" for token in top_tokens])
                st.markdown(token_html, unsafe_allow_html=True)
                
                st.markdown("""
                <br>
                <p>These terms (e.g., *urgent, **verify, **account) are common signals in phishing. **IMMEDIATE QUARANTINE* is recommended.</p>
                """, unsafe_allow_html=True)
            else:
                st.info("No strong keywords found, but the model may have flagged contextual or structural cues.")

        
        # --- LOW RISK Breakdown (Legitimate Keywords) ---
        else: # dashboard_category == 'Safe'
            st.success("✅ Prediction was Legitimate. Model found low risk words.")
            
            # Use the Legitimate keyword finder for Low risk
            top_tokens = get_top_legitimate_tokens(pred_data['text'], cv, model, top_n=8)
            
            if top_tokens:
                st.markdown("##### 🟢 Top Legitimate Keywords Found (Low Risk):")
                
                # Use a safe/green color scheme
                token_html = "".join([f"<span style='background-color: #D4EDDA; color: #155724; padding: 5px 10px; margin: 4px; border-radius: 6px; display: inline-block; font-weight: bold;'>{token}</span>" for token in top_tokens])
                st.markdown(token_html, unsafe_allow_html=True)
                
                st.markdown("""
                <br>
                <p>Keywords (e.g., *agenda, **memo, **review*) align with typical safe communication. Always check the sender's email address to be certain.</p>
                """, unsafe_allow_html=True)
            else:
                st.info("Few distinguishing words were found, but the overall structure was classified as safe.")

            
    else:
        st.info("Please run an analysis on the 'Analyze Email' tab first to see the breakdown here.")
        
    st.markdown("---")
    st.subheader("Common Phishing Tactics")
    st.markdown("""
    - *Urgency/Threats:* Phrases like "Act now," or "Your account will be suspended."
    - *Links/Attachments:* Asking you to click a link to log in or download an unknown file.
    - *Sender Spoofing:* The sender's name looks official, but the actual email address is suspicious (e.g., support@a-mazon.co).
    """)

with tab3:
    history_df = load_analysis()
    render_dashboard(history_df)

with tab4:
    st.subheader("System Architecture and Model Details")
    st.markdown("""
    This application utilizes a simple Machine Learning model for email text classification.
    
    * *Core Task:* Binary Classification (Phishing vs. Legitimate)
    * *Model:* *Multinomial Naive Bayes (MNB)* * *Feature Extraction:* *CountVectorizer* (converts text to word count features).
    * *Data Persistence:* *SQLite3* database (analysis_history.db) is used to store scan results, including metadata like sender and subject, and their action *status* (Pending, Quarantined, Deleted, Marked as Safe).
    
    ### Dashboard Categories Logic
    The dashboard uses a simplified *2-level threat classification*:
    - *Phishing (High - Red):* Model predicted 'Phishing Email' (regardless of confidence).
    - *Safe (Low - Green):* Model predicted 'Legitimate Email'.
    """)
    st.code("""
    # Database Schema
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        sender TEXT,
        subject TEXT,
        prediction TEXT,
        confidence REAL,
        full_email TEXT,
        status TEXT DEFAULT 'Pending' -- Action Status column
    )
    """)
