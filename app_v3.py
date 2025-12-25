import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from datetime import datetime
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Mental Health Assessment", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa; border: 1px solid #dee2e6;
        padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center; padding: 20px; font-size: 12px; color: #666; border-top: 1px solid #eee;
    }
    .emergency-box {
        background-color: #ffebee; border: 2px solid #ef5350; padding: 15px; border-radius: 8px; color: #c62828; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- TRANSLATION DICTIONARY ---
translations = {
    'English': {
        'title': "Student Mental Health Assessment & Risk Prediction",
        'subtitle': "A Machine Learning Based Screening System",
        'reset_btn': "🔄 Reset Form",
        'sidebar_title': "📝 Student Profile",
        'age': "1. Age Group",
        'gender': "2. Gender",
        'uni': "3. University Type",
        'dept': "4. Department",
        'year': "5. Academic Year",
        'cgpa': "6. Current CGPA",
        'scholarship': "7. Scholarship/Waiver?",
        'helpline_title': "🆘 Emergency Helpline (BD)",
        'section_title': "📋 Behavioral Self-Assessment",
        'instructions': "💡 **Instructions:** Please slide the scale to indicate how frequently you have felt these emotions **over the last 2 weeks**.",
        'scale_caption': "Scale: **Not at all** (0) → **Sometimes** (1) → **Often** (2) → **Very Often** (3)",
        'analyze_btn': "🚀 Analyze Risk Level",
        'analyzing': "Machine Learning Model is analyzing...",
        'success': "✅ Assessment Complete",
        'result_title': "📊 Assessment Result",
        'suggestions': "💡 Suggestions",
        'healthy_msg': "🎉 **Status: Healthy**\nYour input pattern suggests a balanced mental state.",
        'download_btn': "📥 Download Assessment Report",
        'disclaimer_short': "⚠️ **Disclaimer:** This is a Machine Learning based screening tool. Results are probabilistic and do not replace professional medical advice.",
        'dev_by': "Developed by",
        'risk_viz': "Risk Visualization",
        'slider_opts': ["Not at all", "Sometimes", "Often", "Very Often"],
        'genders': ['Male', 'Female'],
        'unis': ['Public', 'Private'],
        'scholars': ['Yes', 'No'],
        'years': ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'],
        'depts': ["CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"]
    },
    'Bangla': {
        'title': "শিক্ষার্থী মানসিক স্বাস্থ্য ও ঝুঁকি মূল্যায়ন",
        'subtitle': "মেশিন লার্নিং ভিত্তিক স্ক্রিনিং সিস্টেম",
        'reset_btn': "🔄 ফর্ম রিসেট করুন",
        'sidebar_title': "📝 শিক্ষার্থীর প্রোফাইল",
        'age': "১. বয়স গ্রুপ",
        'gender': "২. লিঙ্গ",
        'uni': "৩. বিশ্ববিদ্যালয়ের ধরণ",
        'dept': "৪. ডিপার্টমেন্ট",
        'year': "৫. শিক্ষাবর্ষ",
        'cgpa': "৬. বর্তমান সিজিপিএ (CGPA)",
        'scholarship': "৭. স্কলারশিপ/ওয়েভার আছে?",
        'helpline_title': "🆘 জরুরি হেল্পলাইন (BD)",
        'section_title': "📋 আচরণগত আত্ম-মূল্যায়ন",
        'instructions': "💡 **নির্দেশনা:** গত **২ সপ্তাহে** আপনি এই অনুভূতিগুলো কতবার অনুভব করেছেন তা স্লাইড করে জানান।",
        'scale_caption': "স্কেল: **একদম না** (০) → **মাঝে মাঝে** (১) → **প্রায়ই** (২) → **খুব বেশি** (৩)",
        'analyze_btn': "🚀 ঝুঁকি বিশ্লেষণ করুন",
        'analyzing': "মেশিন লার্নিং মডেল বিশ্লেষণ করছে...",
        'success': "✅ মূল্যায়ন সম্পন্ন হয়েছে",
        'result_title': "📊 ফলাফল",
        'suggestions': "💡 পরামর্শ",
        'healthy_msg': "🎉 **অবস্থা: সুস্থ**\nআপনার মানসিক অবস্থা ভারসাম্যপূর্ণ মনে হচ্ছে।",
        'download_btn': "📥 রিপোর্ট ডাউনলোড করুন",
        'disclaimer_short': "⚠️ **সতর্কতা:** এটি একটি মেশিন লার্নিং গবেষণা প্রকল্প। এই ফলাফল পেশাদার ডাক্তারি পরামর্শের বিকল্প নয়।",
        'dev_by': "ডেভেলপ করেছে",
        'risk_viz': "ঝুঁকির গ্রাফ",
        'slider_opts': ["একদম না", "মাঝে মাঝে", "প্রায়ই", "খুব বেশি"],
        'genders': ['পুরুষ', 'মহিলা'],
        'unis': ['পাবলিক', 'প্রাইভেট'],
        'scholars': ['হ্যাঁ', 'না'],
        'years': ['১ম বর্ষ', '২য় বর্ষ', '৩য় বর্ষ', '৪র্থ বর্ষ', 'মাস্টার্স'],
        'depts': ["সিএসই", "ইইই", "বিবিএ", "ইংরেজি", "আইন", "ফার্মাসি", "অন্যান্য"]
    }
}

# Questions Translation
q_labels_en = [
    "1. Upset due to academic affairs?", "2. Unable to control important things?", "3. Nervous and stressed?", 
    "4. Could not cope with mandatory activities?", "5. Confident about handling problems?", "6. Things going your way?", 
    "7. Able to control irritations?", "8. Felt academic performance was on top?", "9. Angered due to bad performance?", 
    "10. Difficulties piling up?", "11. Nervous, anxious or on edge?", "12. Unable to stop worrying?", 
    "13. Trouble relaxing?", "14. Being so restless?", "15. Easily annoyed or irritable?", 
    "16. Afraid something awful might happen?", "17. Worrying too much?", "18. Little interest in doing things?", 
    "19. Feeling down, depressed, or hopeless?", "20. Trouble falling/staying asleep?", "21. Feeling tired/little energy?", 
    "22. Poor appetite or overeating?", "23. Feeling bad about yourself?", "24. Trouble concentrating?", 
    "25. Moving slowly or too fast?", "26. Thoughts of hurting yourself?"
]

q_labels_bn = [
    "১. পড়াশোনার চাপে মন খারাপ?", "২. গুরুত্বপূর্ণ বিষয় নিয়ন্ত্রণে অক্ষম?", "৩. নার্ভাস বা মানসিক চাপে ছিলেন?", 
    "৪. বাধ্যতামূলক কাজ সামলাতে পারছেন না?", "৫. সমস্যা সমাধানে আত্মবিশ্বাসী?", "৬. সব কিছু আপনার মতো হচ্ছে?", 
    "৭. বিরক্তি নিয়ন্ত্রণ করতে পারেন?", "৮. একাডেমিক পারফরম্যান্স ভালো মনে হচ্ছে?", "৯. খারাপ ফলাফলে রাগান্বিত?", 
    "১০. সমস্যাগুলো কি পাহাড়সম মনে হচ্ছে?", "১১. উদ্বিগ্ন বা অস্থির অনুভব করেন?", "১২. দুশ্চিন্তা থামাতে পারছেন না?", 
    "১৩. রিল্যাক্স করতে সমস্যা হচ্ছে?", "১৪. খুব বেশি অস্থির লাগছে?", "১৫. সহজেই বিরক্ত হয়ে যাচ্ছেন?", 
    "১৬. ভয় পাচ্ছেন খারাপ কিছু ঘটবে?", "১৭. অতিরিক্ত দুশ্চিন্তা করছেন?", "১৮. কাজে আগ্রহ পাচ্ছেন না?", 
    "১৯. মন খারাপ বা হতাশ লাগছে?", "২০. ঘুমাতে সমস্যা হচ্ছে?", "২১. ক্লান্ত বা শক্তি কম লাগছে?", 
    "২২. ক্ষুধামন্দা বা অতিরিক্ত খাওয়া?", "২৩. নিজেকে নিয়ে খারাপ লাগছে?", "২৪. মনোযোগ দিতে সমস্যা হচ্ছে?", 
    "২৫. খুব ধীর বা দ্রুত নড়াচড়া করছেন?", "২৬. নিজেকে আঘাত করার চিন্তা আসছে?"
]

# --- RESET LOGIC ---
if 'reset' not in st.session_state:
    st.session_state.reset = False

def reset_app():
    st.session_state.reset = True
    st.rerun()

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('mental_health_hybrid_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, encoders, feature_columns
    except Exception as e:
        return None, None, None

model, encoders, feature_columns = load_resources()

if model is None:
    st.error("🚨 Model files missing! Please upload .pkl files to GitHub.")
    st.stop()

# Helper Functions
def extract_number(text):
    try:
        if pd.isna(text): return 0.0
        text_str = str(text)
        if '-' in text_str:
            return float(text_str.split('-')[0].strip())
        match = re.search(r"[-+]?\d*\.\d+|\d+", text_str)
        return float(match.group()) if match else 0.0
    except:
        return 0.0

def get_recommendations(condition, lang):
    tips_en = {
        "Anxiety": ["🌬️ Deep Breathing (4-7-8)", "🧘 Grounding Technique", "☕ Limit Caffeine"],
        "Stress": ["📝 Make To-Do List", "🚶 10-min Walk", "💤 8 Hours Sleep"],
        "Depression": ["🤝 Talk to a Friend", "🌞 Morning Sunlight", "📅 Daily Routine"],
        "Healthy": ["🎉 Keep Going!", "💧 Drink Water", "📖 Journaling"]
    }
    tips_bn = {
        "Anxiety": ["🌬️ গভীর শ্বাস নিন (৪-৭-৮ টেকনিক)", "🧘 গ্রাউন্ডিং এক্সারসাইজ করুন", "☕ ক্যাফেইন (চা/কফি) কমান"],
        "Stress": ["📝 কাজের তালিকা (To-Do List) করুন", "🚶 ১০ মিনিট হাঁটুন", "💤 ৮ ঘণ্টা ঘুমান"],
        "Depression": ["🤝 বন্ধুর সাথে কথা বলুন", "🌞 সকালের রোদে থাকুন", "📅 ছোট রুটিন মেনে চলুন"],
        "Healthy": ["🎉 এভাবেই চালিয়ে যান!", "💧 পর্যাপ্ত পানি পান করুন", "📖 ডায়েরি লিখুন"]
    }
    return tips_bn.get(condition, []) if lang == 'Bangla' else tips_en.get(condition, [])

# --- LANGUAGE SELECTOR ---
st.sidebar.markdown("### 🌐 Language / ভাষা")
lang = st.sidebar.radio("", ('English', 'Bangla'), label_visibility="collapsed")
t = translations[lang]

# --- HEADER ---
col1, col2 = st.columns([8, 2])
with col1:
    st.title(t['title'])
    st.markdown(f"##### {t['subtitle']}")
with col2:
    if st.button(t['reset_btn'], type="primary"):
        reset_app()

st.markdown("---")

# --- SIDEBAR PROFILE ---
st.sidebar.header(t['sidebar_title'])

def get_safe_index(options, default_idx=0):
    if st.session_state.reset: return 0
    return min(default_idx, len(options) - 1)

# --- DYNAMIC MAPPING FOR MODEL ---
# English: Direct | Bangla: Mapped to English

# 1. Age
age_display = ['18-22', '23-26', '27-30', 'Above 30']
age_input = st.sidebar.selectbox(t['age'], age_display, index=get_safe_index(age_display, 0))

# 2. Gender
gender_idx = st.sidebar.selectbox(t['gender'], t['genders'], index=get_safe_index(t['genders'], 0))
gender_model = 'Male' if gender_idx in ['Male', 'পুরুষ'] else 'Female'

# 3. Uni
uni_idx = st.sidebar.selectbox(t['uni'], t['unis'], index=get_safe_index(t['unis'], 1))
uni_model = 'Public' if uni_idx in ['Public', 'পাবলিক'] else 'Private'

# 4. Dept
dept_idx = st.sidebar.selectbox(t['dept'], t['depts'], index=get_safe_index(t['depts'], 0))
dept_map = {
    "সিএসই": "CSE", "ইইই": "EEE", "বিবিএ": "BBA", "ইংরেজি": "English", "আইন": "Law", "ফার্মাসি": "Pharmacy", "অন্যান্য": "Other"
}
dept_model = dept_map.get(dept_idx, dept_idx) # Fallback to input if English

# 5. Year
year_idx = st.sidebar.selectbox(t['year'], t['years'], index=get_safe_index(t['years'], 0))
year_map = {
    '১ম বর্ষ': 'First Year', '২য় বর্ষ': 'Second Year', '৩য় বর্ষ': 'Third Year', '৪র্থ বর্ষ': 'Fourth Year', 'মাস্টার্স': 'Master'
}
year_model = year_map.get(year_idx, year_idx)

# 6. CGPA
cgpa_val = 0.00 if st.session_state.reset else 3.50
cgpa_input = st.sidebar.number_input(t['cgpa'], min_value=0.00, max_value=4.00, value=cgpa_val, step=0.01, format="%.2f")

# 7. Scholarship
sch_idx = st.sidebar.selectbox(t['scholarship'], t['scholars'], index=get_safe_index(t['scholars'], 1))
sch_model = 'Yes' if sch_idx in ['Yes', 'হ্যাঁ'] else 'No'

st.sidebar.markdown("---")

# Helpline
with st.sidebar.expander(t['helpline_title'], expanded=True):
    st.markdown("""
    📞 **Kaan Pete Roi:** 01779554391  
    📞 **Moner Bondhu:** 01779632588  
    🚑 **National Emergency:** 999
    """)

# --- QUESTIONNAIRE ---
st.subheader(t['section_title'])
st.info(t['instructions'])
st.caption(t['scale_caption'])

slider_options = t['slider_opts'] 
# Standardized Mapping: 0-3
options_map = {
    "Not at all": 0, "একদম না": 0,
    "Sometimes": 1, "মাঝে মাঝে": 1,
    "Often": 2, "প্রায়ই": 2,
    "Very Often": 3, "খুব বেশি": 3
}

q_labels = q_labels_bn if lang == 'Bangla' else q_labels_en
answers_map = {}
q_col1, q_col2 = st.columns(2)

for i, q_text in enumerate(q_labels):
    current_col = q_col1 if i % 2 == 0 else q_col2
    with current_col:
        key_name = f"q_{i}_{st.session_state.reset}_{lang}"
        val = st.select_slider(label=f"**{q_text}**", options=slider_options, value=slider_options[0], key=key_name)
        answers_map[i] = options_map[val]
        st.write("") 

final_answers = [answers_map[i] for i in range(26)]

if st.session_state.reset:
    st.session_state.reset = False

# --- PREDICTION ---
st.markdown("---")
col_cen1, col_cen2, col_cen3 = st.columns([1, 2, 1])
with col_cen2:
    analyze_btn = st.button(t['analyze_btn'], type="primary", use_container_width=True)

if analyze_btn:
    age_numeric = extract_number(age_input)
    cgpa_numeric = float(cgpa_input)
    
    # 1. Prepare Input Dictionary (Mapping to English for Model)
    input_dict = {}
    
    # Safety: Ensure we have exactly the columns expected by the model
    if len(feature_columns) == 33:
        input_dict[feature_columns[0]] = age_numeric
        input_dict[feature_columns[1]] = gender_model
        input_dict[feature_columns[2]] = uni_model
        input_dict[feature_columns[3]] = dept_model
        input_dict[feature_columns[4]] = year_model
        input_dict[feature_columns[5]] = cgpa_numeric
        input_dict[feature_columns[6]] = sch_model
        for i in range(26):
            input_dict[feature_columns[7+i]] = final_answers[i]
            
        # 2. DataFrame Creation & Reindexing (CRITICAL FIX: Guarantees Order)
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        try:
            with st.spinner(t['analyzing']):
                probs = model.predict_proba(input_df)
            
            # --- SAFETY CHECK (Q26 Self-Harm) ---
            # Q26 is at index 25. If score >= 2 (Often/Very Often)
            if final_answers[25] >= 2:
                st.markdown(f"""
                <div class="emergency-box">
                    <h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3>
                    <p>{'Your response indicates distress. Please seek professional help or call the helpline immediately.' if lang=='English' else 'আপনার উত্তর মানসিক যন্ত্রণার ইঙ্গিত দিচ্ছে। দয়া করে অবিলম্বে পেশাদার সাহায্য নিন অথবা হেল্পলাইনে কল করুন।'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success(t['success'])
            st.subheader(t['result_title'])
            
            result_cols = st.columns(3)
            conditions = ['Anxiety', 'Stress', 'Depression']
            risk_scores = []
            healthy_count = 0
            
            report_text = f"--- MENTAL HEALTH ASSESSMENT REPORT ---\n"
            report_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_text += f"Profile: {age_input}, {gender_model}, {dept_model}\n"
            report_text += "---------------------------------------\n\n"
            
            for i, cond in enumerate(conditions):
                prob_arr = probs[i][0]
                best_idx = np.argmax(prob_arr)
                label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                confidence = prob_arr[best_idx] * 100
                
                # Display Logic
                display_label = label
                is_healthy = any(safe in label for safe in ["Minimal", "Low", "None", "No Depression"])
                
                # Translate Labels for Display
                if lang == 'Bangla':
                    if is_healthy: display_label = "ঝুঁকি নেই / সুস্থ"
                    elif "Severe" in label: display_label = "তীব্র ঝুঁকি (Severe)"
                    elif "Moderate" in label: display_label = "মাঝারি ঝুঁকি (Moderate)"
                    elif "Mild" in label: display_label = "মৃদু ঝুঁকি (Mild)"
                else:
                    if label == "Minimal Anxiety": display_label = "No Anxiety / Healthy"
                    if label == "Low Stress": display_label = "No Stress / Healthy"
                    if label in ["No Depression", "Minimal Depression"]: display_label = "No Depression / Healthy"

                report_text += f"{cond}: {label} (Confidence: {confidence:.1f}%)\n"
                
                with result_cols[i]:
                    st.markdown(f"#### {cond}")
                    if is_healthy:
                        st.success(f"**{display_label}**")
                        st.progress(0)
                        risk_scores.append((cond, 0))
                        healthy_count += 1
                    else:
                        st.error(f"**{display_label}**")
                        st.progress(int(confidence))
                        st.caption(f"{'Risk Probability' if lang=='English' else 'ঝুঁকির সম্ভাবনা'}: {confidence:.1f}%")
                        risk_scores.append((cond, confidence))

            st.markdown("---")
            
            # --- VISUALIZATION ---
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                st.subheader("📈 " + t['risk_viz'])
                viz_scores = [score if score > 0 else 5 for _, score in risk_scores]
                df_chart = pd.DataFrame({'Condition': conditions, 'Risk Level': viz_scores})
                fig = px.line_polar(df_chart, r='Risk Level', theta='Condition', line_close=True, range_r=[0, 100])
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_v2:
                st.subheader(t['suggestions'])
                if healthy_count == 3:
                    st.balloons()
                    st.markdown(t['healthy_msg'])
                    for tip in get_recommendations("Healthy", lang):
                        st.info(tip)
                else:
                    dominant = max(risk_scores, key=lambda x: x[1])
                    concern_text = "Primary Concern" if lang=='English' else "প্রধান সমস্যা"
                    st.warning(f"🚨 **{concern_text}: {dominant[0]}**")
                    for tip in get_recommendations(dominant[0], lang):
                        st.info(tip)

            # Download Report
            st.markdown("---")
            report_text += "\n---------------------------------------\n"
            report_text += "DISCLAIMER: This result is based on ML patterns and is not a clinical diagnosis."
            
            st.download_button(
                label=t['download_btn'],
                data=report_text,
                file_name=f"Assessment_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Feature column count mismatch!")

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.divider()

st.markdown(f"""
<div class='footer'>
    <div style="font-weight: bold; margin-bottom: 5px;">{t['dev_by']}</div>
    <div style="font-size: 16px; color: #333; font-weight: 600;">Team Dual Core</div>
    <div style="font-size: 14px; margin-top: 5px;">Ovi Sarker & BM Sabbir Hossen Riad</div>
    <div style="font-size: 12px; color: #777; margin-top: 5px;">Department of CSE, Daffodil International University</div>
    <br>
    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; display: inline-block; border: 1px solid #ffeeba;">
        <span style="font-size: 11px; color: #856404;">
            {t['disclaimer_short']}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
