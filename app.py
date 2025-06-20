import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from openai import OpenAI
from dotenv import dotenv_values
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

# settings

st.session_state['mydata'] = None
uploaded_file = None
st.set_page_config(layout="wide", page_title="CSV AI Data Analyzer", page_icon="📊", initial_sidebar_state="expanded")
st.session_state.theme = "dark"
MODEL = "gpt-3.5-turbo"  # Model to use for OpenAI API



# functions

def get_dataframe_summary(df):
    """Generuje podsumowanie dataframe'a dla AI"""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "sample_data": df.head(5).to_dict(),
        "basic_stats": {}
    }
    
    # Dodaj podstawowe statystyki dla kolumn numerycznych
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        try:
            summary["basic_stats"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "null_count": int(df[col].isnull().sum())
            }
        except:
            pass
    
    # Dodaj informacje o kolumnach tekstowych/kategorycznych
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        try:
            summary["basic_stats"][col] = {
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        except:
            pass
    
    return summary

def describe_file_with_openai(df, openai_api_key):
    sample = df.head(100).to_csv(index=False)
    prompt = (
        "Here a data from CSV file. Describe in max 50 words, what the data is about and potential usage\n\n"
        f"{sample}\n\nOpis:"
    )

    # Połączenie z OpenAI API
    openai_client = OpenAI(api_key=openai_api_key)
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a data analyst. Describe the data in the CSV file."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def get_chatbot_reply(user_prompt, memory, openai_api_key, df):
   
    try:
        # Generuj podsumowanie dataframe'a zamiast przekazywać całe dane
        df_summary = get_dataframe_summary(df)
        
        messages = []
        
        # System message z metadanymi o dataframe
        system_message = {
            "role": "system",
            "content": (
                f"You are a data analyst expert using Python and pandas. "
                f"You're analyzing a dataset with the following characteristics:\n\n"
                f"Dataset shape: {df_summary['shape'][0]} rows, {df_summary['shape'][1]} columns\n"
                f"Columns: {', '.join(df_summary['columns'])}\n"
                f"Data types: {df_summary['dtypes']}\n"
                f"Basic statistics: {df_summary['basic_stats']}\n\n"
                f"Sample data (first 5 rows): {df_summary['sample_data']}\n\n"
                "The dataframe is called `df`. "
                "Provide insights, analysis, and answer questions about this data. "
                "If the user asks for a plot, reply with 'PLOT' at the beginning and describe what plot to create."
            )
        }
        
        messages.append(system_message)
        
        # Dodaj ostatnie 10 wiadomości z pamięci (zamiast wszystkich)
        recent_memory = memory[-10:] if len(memory) > 10 else memory
        for message in recent_memory:
            messages.append({"role": message["role"], "content": message["content"]})

        # Dodaj wiadomość użytkownika
        messages.append({"role": "user", "content": user_prompt})
        
        openai_client = OpenAI(api_key=openai_api_key)
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
       
        return {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Przepraszam, wystąpił błąd podczas analizy: {str(e)}",
        }

# Main

st.title("CSV Data Analyzer")
st.subheader("for imprv.ai,  v1.0 by label_it")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# sidebar

with st.sidebar:
    # File uploader
    uploaded_file = st.sidebar.file_uploader("", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=";")
            st.session_state['mydata'] = df
            st.success(f"File uploaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            try:
                df = pd.read_csv(uploaded_file, sep=",")
                st.session_state['mydata'] = df
                st.success(f"File uploaded successfully with comma separator! {df.shape[0]} rows, {df.shape[1]} columns")
            except:
                st.error("Could not read the file. Please check the format.")
    else:
        st.info("Please upload a CSV file to start analyzing your data.")   
  
    # Filters
    if uploaded_file is not None and st.session_state['mydata'] is not None:
        with st.expander("Expand to see Smart filters."):
            # search input 
            filtered_df = st.session_state['mydata'].copy()
            search_term = st.text_input("Search in data", "")

            for filter_col in st.session_state['mydata'].columns:
                if st.checkbox(f"Show {filter_col} filter", value=False):
                    unique_values = sorted(st.session_state['mydata'][filter_col].dropna().unique())
                    selected_values = st.multiselect(f"Select values for {filter_col}", unique_values)
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[filter_col].isin(selected_values)]
            
            if search_term:
                mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)
                filtered_df = filtered_df[mask]
            
            # Update session state with filtered data
            st.session_state['filtered_data'] = filtered_df

    # AI description
    if uploaded_file is not None and st.session_state['mydata'] is not None:
        # OpenAI API key input
        openai_api_key = st.text_input("Please add your OpenAI API key:", type="password")
        if openai_api_key:
            st.success("API Key loaded.")
        else:
            st.info("Enter your OpenAI API key to use AI features.")
    
        if st.button("Show AI description") and openai_api_key:
            with st.spinner("Generating AI description..."):
                description = describe_file_with_openai(st.session_state['mydata'], openai_api_key)
                st.write(description)

if uploaded_file is not None and st.session_state['mydata'] is not None:
    # Use filtered data if available, otherwise use original data
    display_df = st.session_state.get('filtered_data', st.session_state['mydata'])
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data explorer", "YData Profiling Report", "Chart generator", "Data chatter"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Number of rows", display_df.shape[0])
        with col2:
            st.metric("Number of columns", display_df.shape[1])
        with col3:
            missing_percentage = (display_df.isnull().sum().sum() / (display_df.shape[0] * display_df.shape[1]) * 100)
            st.metric("Missing data %", f"{missing_percentage:.1f}%")
        with col4:
            memory_usage = display_df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory usage", f"{memory_usage:.1f} MB")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=800)
    
    with tab2:
        st.write("YData Profiling Report")
        if st.button("Generate Profile Report"):
            with st.spinner("Generating profile report... This may take a while for large datasets."):
                try:
                    # For large datasets, use minimal profile
                    sample_size = min(1000, len(display_df))
                    sample_df = display_df.sample(n=sample_size) if len(display_df) > sample_size else display_df
                    
                    profile = ProfileReport(sample_df, minimal=True, title="Dataset Profile Report")
                    profile_html = profile.to_html()
                    html(profile_html, height=1000, scrolling=True)
                    
                    if len(display_df) > sample_size:
                        st.info(f"Report generated using a sample of {sample_size} rows from {len(display_df)} total rows.")
                except Exception as e:
                    st.error(f"Error generating profile report: {str(e)}")

    with tab3:
        st.subheader("Chart Generator")
        
        # Chart configuration
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            chart_type = st.selectbox("Select chart type", ["Bar", "Line", "Scatter", "Histogram", "Box Plot", "Violin Plot"])
        with col2:
            x_axis = st.selectbox("Select X axis", display_df.columns)
        with col3:
            if chart_type not in ["Histogram"]:
                y_axis = st.selectbox("Select Y axis", display_df.columns)
            else:
                y_axis = None
        with col4:
            chart_theme = st.selectbox("Chart theme", ["darkgrid", "whitegrid", "dark", "white", "ticks"])
        
        # Additional chart options
        col5, col6, col7 = st.columns(3)
        with col5:
            figure_size = st.selectbox("Figure size", ["Small (8x5)", "Medium (10x6)", "Large (12x8)", "Extra Large (14x10)"])
        with col6:
            label_rotation = st.slider("X-axis label rotation", 0, 90, 45, step=15)
        with col7:
            max_labels = st.slider("Max X-axis labels", 5, 50, 20)
        
        if st.button("Generate chart"):
            with st.spinner("Generating chart..."):
                try:
                    # Set seaborn theme
                    sns.set_theme(style=chart_theme)
                    
                    # Parse figure size
                    size_map = {
                        "Small (8x5)": (8, 5),
                        "Medium (10x6)": (10, 6),
                        "Large (12x8)": (12, 8),
                        "Extra Large (14x10)": (14, 10)
                    }
                    fig_size = size_map[figure_size]
                    
                    plt.figure(figsize=fig_size)
                    
                    # Sample data if too large for plotting
                    plot_df = display_df.sample(n=min(1000, len(display_df))) if len(display_df) > 1000 else display_df
                    
                    # Handle too many unique values on x-axis
                    if chart_type in ["Bar"] and plot_df[x_axis].nunique() > max_labels:
                        # Get top N most frequent values
                        top_values = plot_df[x_axis].value_counts().head(max_labels).index
                        plot_df = plot_df[plot_df[x_axis].isin(top_values)]
                        st.warning(f"Showing only top {max_labels} most frequent values for {x_axis} to improve readability.")
                    
                    # Create plots
                    if chart_type == "Bar":
                        if plot_df[x_axis].dtype in ['object', 'category']:
                            # For categorical data, aggregate by mean
                            if y_axis and plot_df[y_axis].dtype in ['int64', 'float64']:
                                plot_data = plot_df.groupby(x_axis)[y_axis].mean().reset_index()
                                sns.barplot(data=plot_data, x=x_axis, y=y_axis)
                            else:
                                sns.countplot(data=plot_df, x=x_axis)
                        else:
                            sns.barplot(data=plot_df, x=x_axis, y=y_axis)
                    elif chart_type == "Line":
                        sns.lineplot(data=plot_df, x=x_axis, y=y_axis)
                    elif chart_type == "Scatter":
                        sns.scatterplot(data=plot_df, x=x_axis, y=y_axis, alpha=0.7)
                    elif chart_type == "Histogram":
                        sns.histplot(data=plot_df, x=x_axis, bins=30)
                    elif chart_type == "Box Plot":
                        sns.boxplot(data=plot_df, x=x_axis, y=y_axis)
                    elif chart_type == "Violin Plot":
                        sns.violinplot(data=plot_df, x=x_axis, y=y_axis)
                    
                    # Improve title and labels
                    chart_title = f"{chart_type} chart: {x_axis}" + (f" vs {y_axis}" if y_axis else "")
                    plt.title(chart_title, fontsize=14, fontweight='bold', pad=20)
                    plt.xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
                    if y_axis:
                        plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
                    
                    # Handle x-axis labels
                    ax = plt.gca()
                    
                    # Rotate labels
                    plt.xticks(rotation=label_rotation, ha='right' if label_rotation > 45 else 'center')
                    
                    # Limit number of x-axis labels if too many
                    if len(ax.get_xticklabels()) > max_labels:
                        # Show every nth label
                        n = len(ax.get_xticklabels()) // max_labels + 1
                        for i, label in enumerate(ax.get_xticklabels()):
                            if i % n != 0:
                                label.set_visible(False)
                    
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()
                    
                    # Add some padding to bottom if labels are rotated
                    if label_rotation > 0:
                        plt.subplots_adjust(bottom=0.15)
                    
                    # Display chart
                    st.pyplot(plt)
                    plt.clf()  # Clear the figure after displaying it
                    
                    if len(display_df) > 1000:
                        st.info(f"Chart generated using a sample of {len(plot_df)} rows from {len(display_df)} total rows.")
                        
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
                    st.error("Try selecting different columns or chart type.")
                finally:
                    # Reset to default theme
                    sns.set_theme()
    
    with tab4:
        st.subheader("Data Chatter - AI Assistant")
        
        # OpenAI API key check
        if 'openai_api_key' not in locals() or not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to use the chat feature.")
        else:
            # Clear chat button at the top
            if st.button("Clear Chat History"):
                st.session_state["messages"] = []
                st.rerun()
            
            # Container for chat messages
            chat_container = st.container()
            
            # Display chat messages in container
            with chat_container:
                for message in st.session_state["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Chat input at the bottom
            prompt = st.chat_input("What do you want to ask about the data?")
            
            if prompt:
                # Add user message to chat
                user_message = {"role": "user", "content": prompt}
                st.session_state["messages"].append(user_message)

                # Get AI response
                with st.spinner("Analyzing data..."):
                    chatbot_message = get_chatbot_reply(
                        prompt, 
                        memory=st.session_state["messages"], 
                        openai_api_key=openai_api_key, 
                        df=display_df
                    )
                    
                    # Check if response contains PLOT request
                    if chatbot_message["content"].startswith("PLOT"):
                        chatbot_message["content"] += "\n\n💡 *Tip: You can use the Chart Generator tab to create visualizations.*"
                
                # Add assistant message to chat
                st.session_state["messages"].append(chatbot_message)
                
                # Rerun to display new messages
                st.rerun()
else:
    st.info("Please upload a CSV file to start analyzing your data.")