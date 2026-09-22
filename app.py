import numpy as np
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Employee Performance Predictor")
st.write(
    "Enter an employee's **Training Hours** and **Attendance (%)** to predict "
    "whether their performance is **Good** or **Needs Improvement**."
)

st.divider()



@st.cache_resource
def load_ann_model():
  """Loads and caches the trained Keras model so it isn't reloaded on every rerun."""
  return tf.keras.models.load_model("employee_performance_ann.keras")


try:
  model = load_ann_model()
except Exception as e:
  st.error(
      "⚠️ Could not load `employee_performance_ann.keras`. "
      "Please make sure you have run the training script first and the model file "
      "is located in the same directory as `app.py`."
  )
  st.stop()


col1, col2 = st.columns(2)

with col1:
  training_hours = st.number_input(
      "Training Hours",
      min_value=0.0,
      max_value=40.0,
      value=8.0,
      step=1.0,
      help="Number of training hours completed by the employee.",
  )

with col2:
  attendance = st.number_input(
      "Attendance (%)",
      min_value=0.0,
      max_value=100.0,
      value=75.0,
      step=1.0,
      help="Attendance percentage of the employee.",
  )

st.write("")

if st.button("Predict Performance", type="primary", use_container_width=True):
  # Prepare 2D input array for ANN: shape (1, 2)
  input_data = np.array([[training_hours, attendance]], dtype=np.float32)

  # Model forward pass
  probability = float(model.predict(input_data, verbose=0)[0][0])

  # Binary classification threshold
  result = "Good" if probability >= 0.5 else "Needs Improvement"

  st.subheader("Prediction Result")

  if result == "Good":
    st.success("✅ **Performance Result: GOOD**")
  else:
    st.warning("⚠️ **Performance Result: NEEDS IMPROVEMENT**")

  # Metrics display
  m_col1, m_col2, m_col3 = st.columns(3)
  m_col1.metric("Training Hours", f"{training_hours:.1f} hrs")
  m_col2.metric("Attendance", f"{attendance:.1f}%")
  m_col3.metric("Good Probability", f"{probability * 100:.2f}%")

  # Confidence progress bar
  st.caption("Model Confidence (Probability of 'Good' performance):")
  st.progress(probability)
