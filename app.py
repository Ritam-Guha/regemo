import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Multi-Objective Optimization Dashboard")

# Input fields for hyperparameters
param1 = st.number_input("Hyperparameter 1", value=1.0)
param2 = st.number_input("Hyperparameter 2", value=2.0)

# Function to simulate running the optimization
def run_optimization(param1, param2):
    # Example data
    data = {
        'Objective1': [1, 2, 3, 4, 5],
        'Objective2': [2, 4, 1, 3, 5],
        'Parameter': [param1, param2, param1 + param2, param1 * 2, param2 * 2],
        'Category': ['A', 'B', 'C', 'A', 'B']
    }
    df = pd.DataFrame(data)
    df.to_csv('results.csv', index=False)

# Run optimization when button is clicked
if st.button('Run Optimization'):
    run_optimization(param1, param2)
    st.success("Optimization completed!")

# Show results and plots when button is clicked
if st.button('Show Results'):
    df = pd.read_csv('results.csv')
    st.write("### Optimization Results")
    st.dataframe(df)

    # Scatter Plot
    st.write("### Scatter Plot")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Objective1'], df['Objective2'], c=df['Parameter'], cmap='viridis')
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    plt.colorbar(scatter, label='Parameter')
    st.pyplot(fig)

    # Line Plot
    st.write("### Line Plot")
    fig, ax = plt.subplots()
    ax.plot(df['Objective1'], df['Objective2'], marker='o')
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    st.pyplot(fig)

    # Heatmap
    st.write("### Heatmap")
    heatmap_data = df.pivot("Objective1", "Objective2", "Parameter")
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # Bar Chart
    st.write("### Bar Chart")
    fig, ax = plt.subplots()
    df.groupby('Category')['Parameter'].sum().plot(kind='bar', ax=ax)
    ax.set_ylabel("Sum of Parameters")
    st.pyplot(fig)