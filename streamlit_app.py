import streamlit as st
from sklearn.linear_model import LinearRegression
import pandas as pd


def main():
    # Function to connect to a public Google Sheet by URL
    sheet_url = "https://docs.google.com/spreadsheets/d/1lYhQeVP-9Ts_cSKiWYnQi_YDb_HEUF8VsXvgmdnDjEY/export?format=csv"

    # Define the data types for each column (excluding Duration for now)
    dtype_dict = {
        "Distance": "float",
        "Price": "float",
    }

    # Read the CSV into a DataFrame with specified data types
    raw_df = pd.read_csv(sheet_url, dtype=dtype_dict, parse_dates=["Timestamp"])
    df = raw_df.copy()
    raw_df = raw_df.drop(columns=["Comment (optional)"])

    # Convert the Duration column to timedelta
    df['Duration'] = pd.to_timedelta(df['Duration'])

    # Create a column with the duration in minutes
    df['Duration_minutes'] = df['Duration'].dt.total_seconds() / 60

    # Perform multiple linear regression
    X1 = df[['Duration_minutes', 'Distance']]
    y1 = df['Price']
    model = LinearRegression()
    model.fit(X1, y1)

    # Create a model for each city
    area_list = df['Area'].unique()
    city_models = {}
    for area in area_list:
        X_area = df[df['Area'] == area][['Duration_minutes', 'Distance']]
        y_area = df[df['Area'] == area]['Price']
        model_area = LinearRegression()
        model_area.fit(X_area, y_area)
        city_models[area] = model_area


    ### STREAMLIT APP ###
    st.title("🚗 Waymo Price Tracker 💰")
    st.write("_An open-source, open-data price tracker for Waymo one rides._")
    st.markdown("- Submit your rides: https://forms.gle/SAakkvWg5FB2tzLc6\n"
                "- Git repo: https://github.com/EwoutH/Waymo-pricing")

    # Display raw data table
    st.subheader("Raw data")
    st.dataframe(raw_df)

    # Display the model coefficients
    st.subheader("General model coefficients")
    st.write(f"Price = \\${model.intercept_:.2f} + \\${model.coef_[0]:.2f} per mile + \\${model.coef_[1]:.2f} per minute")
    st.subheader(f"Area models coefficients")
    st.write(f"_Coming soon with more data!_")

    # Scatter plots for distance vs. price and duration vs. price
    st.subheader("Scatter plots")

    col1, col2 = st.columns(2)

    with col1:
        st.scatter_chart(df, x='Distance', y='Price', color='Area')

    with col2:
        st.scatter_chart(df, x='Duration_minutes', y='Price', color='Area')

    # Input form for price prediction
    st.subheader("Predict Price")
    with st.form("input_form"):
        duration_input = st.number_input("Enter duration in minutes:", min_value=0.0, value=30.0)
        distance_input = st.number_input("Enter distance in miles:", min_value=0.0, value=5.0)
        city_input = st.selectbox("Select area:", options=["Los Angeles", "San Francisco", "Phoenix"])
        submit_button = st.form_submit_button("Predict Price")

        if submit_button:
            gen_prediction = model.predict([[duration_input, distance_input]])[0]
            if city_input:
                city_prediction = city_models[city_input].predict([[duration_input, distance_input]])[0]

            st.write(f"Predicted general price: \\${gen_prediction:.2f}")
            st.write(f"Predicted price for {city_input}: \\${city_prediction:.2f}")

if __name__ == "__main__":
    main()
