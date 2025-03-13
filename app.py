import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#LOADING DATA FROM GOOGLE SHEETS
st.title("📊 Doctor's Time")

st.write(
    "This web app predicts the best doctors (NPIs) to send survey invitations based on their login time."
)

sheet_url = "https://docs.google.com/spreadsheets/d/1FnH0XF4UdpsP7Tz6bG3_NIFOyD-HCQK6/export?format=csv"

@st.cache_data
def load_data():
    df = pd.read_csv(sheet_url)
    
    # Convert time columns
    df["Login Time"] = pd.to_datetime(df["Login Time"], errors="coerce")
    df = df.dropna(subset=["Login Time"])  # Remove invalid dates
    df["Login Hour"] = df["Login Time"].dt.hour
    
    # Encode categorical variables
    le_speciality = LabelEncoder()
    df["Speciality Encoded"] = le_speciality.fit_transform(df["Speciality"])

    le_region = LabelEncoder()
    df["Region Encoded"] = le_region.fit_transform(df["Region"])
    
    return df

df = load_data()

# TRAINING ML MODEL
features = ["Login Hour", "Usage Time (mins)", "Speciality Encoded", "Region Encoded", "Count of Survey Attempts"]
target = "NPI"

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# USER INPUT FOR TIME PREDICTION
st.write("### ⏳ Enter a Time to Know Active Doctors")
user_time = st.slider("Select an Hour (0-23)", min_value=0, max_value=23, value=12)

# Filter doctors who are active at the selected time
recommended_doctors = df[df["Login Hour"] == user_time][["NPI", "State", "Speciality", "Region"]]

# Show results
if not recommended_doctors.empty:
    st.write(f"#### 👨‍⚕️ Doctors Active at {user_time}:00")
    st.dataframe(recommended_doctors)
    
    # Convert to CSV
    csv_data = recommended_doctors.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv_data, file_name="recommended_doctors.csv", mime="text/csv")
else:
    st.warning("⚠️ No doctors found for this time slot.")
