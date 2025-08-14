import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="APS360 NEOFFI Model Predictions",
    page_icon="🧠",
    layout="wide"
)

def get_subjects():
    """Get available subjects for dropdown"""
    try:
        from load import get_available_subjects
        subjects = get_available_subjects()
        return subjects
    except Exception as e:
        st.error(f"❌ Error loading subjects: {e}")
        return []

def predict_single_subject(subject_idx):
    """Make prediction for a single subject"""
    try:
        from load import load_model_and_predict_single
        
        with st.spinner(f"Running prediction for subject {subject_idx + 1}..."):
            gt, pred, subject_id, traits = load_model_and_predict_single(subject_idx)
            
        return {
            'ground_truth': gt,
            'prediction': pred,
            'subject_id': subject_id,
            'traits': traits,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_radar_chart(gt, pred, traits, subject_id):
    """Create radar chart for personality profile"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=gt,
        theta=traits,
        fill='toself',
        name='Ground Truth',
        line_color='blue',
        fillcolor='rgba(0,0,255,0.1)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=pred,
        theta=traits,
        fill='toself',
        name='Prediction',
        line_color='red',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(gt), max(pred)) * 1.1]
            )
        ),
        showlegend=True,
        title=f"Personality Profile - {subject_id}",
        height=500
    )
    
    return fig

def create_bar_chart(gt, pred, traits, subject_id):
    """Create bar chart comparison"""
    df = pd.DataFrame({
        'Trait': traits + traits,
        'Value': list(gt) + list(pred),
        'Type': ['Ground Truth'] * len(traits) + ['Prediction'] * len(traits)
    })
    
    fig = px.bar(
        df,
        x='Trait',
        y='Value',
        color='Type',
        barmode='group',
        title=f"Ground Truth vs Predictions for {subject_id}",
        height=400
    )
    
    return fig

def main():
    st.title("🧠 APS360 NEOFFI Model Predictions Dashboard")
    st.markdown("---")
    
    # Load available subjects
    subjects = get_subjects()
    
    if not subjects:
        st.error("❌ No subjects available. Please check your data paths and ensure the model files exist.")
        st.stop()
    
    # Subject selection
    st.sidebar.title("Subject Selection")
    st.sidebar.markdown("Select a subject to analyze:")
    
    subject_options = [s['display_name'] for s in subjects]
    selected_subject_name = st.sidebar.selectbox(
        "Choose subject:",
        subject_options,
        key="subject_selector"
    )
    
    # Get selected subject index
    selected_subject_idx = next(
        (s['index'] for s in subjects if s['display_name'] == selected_subject_name),
        0
    )
    
    # Predict button
    if st.sidebar.button("🔮 Run Prediction", type="primary"):
        # Store prediction in session state
        result = predict_single_subject(selected_subject_idx)
        st.session_state['prediction_result'] = result
        st.session_state['selected_subject'] = selected_subject_name
    
    # Display results if available
    if 'prediction_result' in st.session_state and st.session_state['prediction_result']['success']:
        result = st.session_state['prediction_result']
        gt = result['ground_truth']
        pred = result['prediction']
        subject_id = result['subject_id']
        traits = result['traits']
        
        # Main content area
        st.header(f"📊 Analysis Results for {st.session_state.get('selected_subject', 'Selected Subject')}")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        mae = np.mean(np.abs(gt - pred))
        rmse = np.sqrt(np.mean((gt - pred) ** 2))
        correlation = np.corrcoef(gt, pred)[0, 1] if len(gt) > 1 else 0
        max_error = np.max(np.abs(gt - pred))
        
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.3f}")
        with col3:
            st.metric("Correlation", f"{correlation:.3f}")
        with col4:
            st.metric("Max Error", f"{max_error:.3f}")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Personality Radar Chart")
            radar_fig = create_radar_chart(gt, pred, traits, subject_id)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Trait Comparison")
            bar_fig = create_bar_chart(gt, pred, traits, subject_id)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📈 Detailed Results")
        
        results_df = pd.DataFrame({
            'Trait': traits,
            'Ground Truth': gt,
            'Prediction': pred,
            'Absolute Error': np.abs(gt - pred),
            'Relative Error (%)': np.abs((gt - pred) / (gt + 1e-8)) * 100
        })
        
        st.dataframe(
            results_df.round(3),
            use_container_width=True,
            hide_index=True
        )
        
        # Interpretation section
        st.subheader("🔍 Interpretation")
        
        # Find best and worst predictions
        errors = np.abs(gt - pred)
        best_trait_idx = np.argmin(errors)
        worst_trait_idx = np.argmax(errors)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Best Prediction:** {traits[best_trait_idx]}")
            st.write(f"- Ground Truth: {gt[best_trait_idx]:.3f}")
            st.write(f"- Prediction: {pred[best_trait_idx]:.3f}")
            st.write(f"- Error: {errors[best_trait_idx]:.3f}")
        
        with col2:
            st.warning(f"**Largest Error:** {traits[worst_trait_idx]}")
            st.write(f"- Ground Truth: {gt[worst_trait_idx]:.3f}")
            st.write(f"- Prediction: {pred[worst_trait_idx]:.3f}")
            st.write(f"- Error: {errors[worst_trait_idx]:.3f}")
        
        # Trait explanations
        st.subheader("📚 Trait Explanations")
        trait_explanations = {
            'NEOFAC_A': 'Agreeableness - Tendency to be compassionate and cooperative',
            'NEOFAC_O': 'Openness - Openness to experience and intellectual curiosity', 
            'NEOFAC_C': 'Conscientiousness - Tendency to be organized and dependable',
            'NEOFAC_N': 'Neuroticism - Tendency to experience negative emotions',
            'NEOFAC_E': 'Extraversion - Tendency to seek stimulation in the company of others'
        }
        
        for trait, explanation in trait_explanations.items():
            if trait in traits:
                idx = traits.index(trait)
                st.write(f"**{trait}:** {explanation}")
                st.write(f"  - Score: {gt[idx]:.3f} (actual) vs {pred[idx]:.3f} (predicted)")
    
    elif 'prediction_result' in st.session_state and not st.session_state['prediction_result']['success']:
        # Show error if prediction failed
        st.error(f"❌ Prediction failed: {st.session_state['prediction_result']['error']}")
    
    else:
        # Show instructions when no prediction has been run
        st.info("👈 Select a subject from the sidebar and click 'Run Prediction' to analyze")
        
        # Show available subjects
        st.subheader("📋 Available Subjects")
        subjects_df = pd.DataFrame(subjects)
        if not subjects_df.empty:
            st.dataframe(
                subjects_df[['display_name', 'id']].rename(columns={
                    'display_name': 'Display Name',
                    'id': 'Subject ID'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("*APS360 NEOFFI Project - Model Predictions Dashboard*")
    
    # Debug info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("Debug Info")
        st.write(f"Available subjects: {len(subjects)}")
        if subjects:
            st.write("Subject IDs:")
            for s in subjects:
                st.write(f"- {s['id']}")

if __name__ == "__main__":
    main()