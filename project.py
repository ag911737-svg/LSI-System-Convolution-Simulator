import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="LSI System Simulator",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 LSI System & Convolution Simulator")
st.markdown(
    "### Interactive Signal Processing Application"
)
st.write(
    "Generate input signals, impulse responses "
    "and visualize their convolution."
)

st.divider()

# Sidebar Controls
st.sidebar.header("⚙️ Signal Controls")

signal_type = st.sidebar.selectbox(
    "Select Input Signal",
    ["Unit Step", "Unit Impulse", "Exponential", "Rectangular"]
)

response_type = st.sidebar.selectbox(
    "Select Impulse Response",
    ["Exponential", "Unit Step", "Unit Impulse"]
)

amplitude = st.sidebar.slider(
    "Amplitude", 0.5, 5.0, 1.0
)

decay = st.sidebar.slider(
    "Decay Factor", 0.1, 0.9, 0.5
)

N = st.sidebar.slider(
    "Number of Samples", 10, 50, 20
)

# Generate Input Signal
n = np.arange(N)

if signal_type == "Unit Step":
    x = amplitude * np.ones(N)

elif signal_type == "Unit Impulse":
    x = np.zeros(N)
    x[0] = amplitude

elif signal_type == "Exponential":
    x = amplitude * (decay ** n)

else:
    x = np.zeros(N)
    x[:N // 2] = amplitude

# Generate Impulse Response
if response_type == "Exponential":
    h = decay ** n

elif response_type == "Unit Step":
    h = np.ones(N)

else:
    h = np.zeros(N)
    h[0] = 1

# Convolution
y = np.convolve(x, h)
ny = np.arange(len(y))

# Display Signals
st.subheader("📈 Signal Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("Input Signal x[n]")
    fig1, ax1 = plt.subplots()
    ax1.stem(n, x)
    ax1.set_xlabel("Time (n)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.write("Impulse Response h[n]")
    fig2, ax2 = plt.subplots()
    ax2.stem(n, h)
    ax2.set_xlabel("Time (n)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    st.pyplot(fig2)
    plt.close(fig2)

# Output Signal
st.subheader("🔄 Convolution Output y[n]")

fig3, ax3 = plt.subplots()
ax3.stem(ny, y)
ax3.set_xlabel("Time (n)")
ax3.set_ylabel("Amplitude")
ax3.set_title("Output Signal")
ax3.grid(True)
st.pyplot(fig3)
plt.close(fig3)

# Mathematical Formula
st.latex(r"y[n] = x[n] * h[n]")

st.success("Convolution calculated successfully!")

# Download CSV
data = pd.DataFrame({
    "Input x[n]": pd.Series(x),
    "Impulse h[n]": pd.Series(h),
    "Output y[n]": pd.Series(y)
})

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Signal Data (CSV)",
    data=csv,
    file_name="signal_data.csv",
    mime="text/csv"
)
# =====================================
# SYSTEM ANALYSIS
# =====================================

st.divider()
st.header("System Analysis")

# Step Response
st.subheader("1. Step Response")

step_input = np.ones(N)

step_output = np.convolve(step_input, h)

fig4, ax4 = plt.subplots()

ax4.stem(np.arange(len(step_output)), step_output)

ax4.set_title("Unit Step Response")
ax4.set_xlabel("Time (n)")
ax4.set_ylabel("Amplitude")
ax4.grid(True)

st.pyplot(fig4)
plt.close(fig4)

# Causality Check
st.subheader("2. Causality Analysis")

st.success(
    "System is Causal because the impulse response "
    "is defined only for n >= 0."
)

# Stability Check
st.subheader("3. Stability Analysis")

if response_type == "Unit Step":
    st.error("System is Unstable (BIBO).")

else:
    st.success("System is Stable (BIBO).")

# System Information
st.subheader("4. System Information")

col1, col2, col3 = st.columns(3)

col1.metric("Input Samples", N)
col2.metric("Output Samples", len(y))
col3.metric("Impulse Response", response_type)

st.info(
    "The output is calculated using discrete-time "
    "linear convolution."
)
# =====================================
# CONTINUOUS & DISCRETE TIME SIGNALS
# =====================================

st.divider()
st.header("Continuous-Time & Discrete-Time Signals")

signal_choice = st.selectbox(
    "Select Signal for CT/DT Visualization",
    ["Sine", "Cosine", "Square", "Exponential"]
)

amplitude_ct = st.slider(
    "Select Amplitude", 0.5, 5.0, 1.0,
    key="amp_ct"
)

frequency_ct = st.slider(
    "Select Frequency", 0.1, 5.0, 1.0,
    key="freq_ct"
)

# Continuous Time
t = np.linspace(-2, 2, 500)

# Discrete Time
n_ct = np.arange(-10, 11)

if signal_choice == "Sine":
    ct = amplitude_ct * np.sin(2*np.pi*frequency_ct*t)
    dt = amplitude_ct * np.sin(2*np.pi*frequency_ct*n_ct/10)

elif signal_choice == "Cosine":
    ct = amplitude_ct * np.cos(2*np.pi*frequency_ct*t)
    dt = amplitude_ct * np.cos(2*np.pi*frequency_ct*n_ct/10)

elif signal_choice == "Square":
    from scipy import signal
    ct = amplitude_ct * signal.square(
        2*np.pi*frequency_ct*t
    )
    dt = amplitude_ct * signal.square(
        2*np.pi*frequency_ct*n_ct/10
    )

else:
    ct = amplitude_ct * np.exp(-frequency_ct*np.maximum(t, 0))
    dt = amplitude_ct * np.exp(
        -frequency_ct*np.maximum(n_ct, 0)/10
    )

# Plot Continuous and Discrete Signals

col1, col2 = st.columns(2)

with col1:
    st.subheader("Continuous-Time Signal")

    fig_ct, ax_ct = plt.subplots()
    ax_ct.plot(t, ct)
    ax_ct.set_xlabel("Time (t)")
    ax_ct.set_ylabel("Amplitude")
    ax_ct.grid(True)
    st.pyplot(fig_ct)
    plt.close(fig_ct)

with col2:
    st.subheader("Discrete-Time Signal")

    fig_dt, ax_dt = plt.subplots()
    ax_dt.stem(n_ct, dt)
    ax_dt.set_xlabel("Time (n)")
    ax_dt.set_ylabel("Amplitude")
    ax_dt.grid(True)
    st.pyplot(fig_dt)
    plt.close(fig_dt)

st.success("CT and DT signals generated successfully!")
# =====================================
# TIME SHIFTING, SCALING & PHASE
# =====================================

st.divider()
st.header("Time Shifting, Scaling & Phase Adjustment")

from scipy import signal

signal_type = st.selectbox(
    "Choose Signal",
    ["Sine", "Cosine", "Square", "Triangular"],
    key="transform_signal"
)

A = st.slider("Amplitude", 0.5, 5.0, 1.0, key="trans_amp")
f = st.slider("Frequency", 0.5, 5.0, 1.0, key="trans_freq")

shift = st.slider(
    "Time Shift (seconds)", -2.0, 2.0, 0.0,
    key="time_shift"
)

scale = st.slider(
    "Time Scaling Factor", 0.5, 3.0, 1.0,
    key="time_scale"
)

phase = st.slider(
    "Phase (radians)", 0.0, 6.28, 0.0,
    key="phase_angle"
)

t_transform = np.linspace(-3, 3, 1000)

# Original Signal
def create_signal(t):
    if signal_type == "Sine":
        return A * np.sin(2*np.pi*f*t)

    elif signal_type == "Cosine":
        return A * np.cos(2*np.pi*f*t)

    elif signal_type == "Square":
        return A * signal.square(2*np.pi*f*t)

    else:
        return A * signal.sawtooth(
            2*np.pi*f*t, width=0.5
        )

original = create_signal(t_transform)

# Modified Signal: x(a(t-shift)) with phase
modified = A * np.sin(
    2*np.pi*f*scale*(t_transform-shift) + phase
) if signal_type == "Sine" else (
    A * np.cos(2*np.pi*f*scale*(t_transform-shift) + phase)
    if signal_type == "Cosine"
    else create_signal(scale*(t_transform-shift))
)

# Graph Comparison
fig_transform, ax = plt.subplots(figsize=(10, 4))

ax.plot(t_transform, original, label="Original Signal")
ax.plot(t_transform, modified, label="Modified Signal")

ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.set_title("Signal Transformation")
ax.legend()
ax.grid(True)

st.pyplot(fig_transform)
plt.close(fig_transform)

st.latex(r"x(at-b)")

st.success("Signal transformation completed!")
# =====================================
# PROFESSIONAL UI DESIGN
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #17365D;
    font-weight: 800;
}

h2, h3 {
    color: #24476B;
}

[data-testid="stSidebar"] {
    background-color: #eaf0f8;
}

.stButton>button {
    background-color: #2457A7;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 📡 LSI System Simulator
### Interactive Signals and Systems Analysis

Explore Linear Shift-Invariant Systems through
interactive simulation, signal visualization and analysis.

---
""")