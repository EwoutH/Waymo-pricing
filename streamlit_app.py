import streamlit as st
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import altair as alt



def prediction_interval(x, y, degree, minX, maxY, confidence=0.95):
    p, cov = np.polyfit(x, y, degree, cov=True)
    y_model = np.polyval(p, x)
    n = len(y)
    m = p.size
    dof = n - m
    t = stats.t.ppf(1 - (1 - confidence) / 2., dof)
    resid = y - y_model
    chi2 = np.sum((resid / np.std(resid))**2)
    s_err = np.sqrt(np.sum(resid**2) / dof)
    x2 = np.linspace(minX, maxY, 100)
    y2 = np.polyval(p, x2)
    ci = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return x2, y2, y2 - ci, y2 + ci


def main():
    # Function to connect to a public Google Sheet by URL
    sheet_url = "https://docs.google.com/spreadsheets/d/1lYhQeVP-9Ts_cSKiWYnQi_YDb_HEUF8VsXvgmdnDjEY/export?format=csv"
    # sheet_url = "https://docs.google.com/spreadsheets/d/1lYhQeVP-9Ts_cSKiWYnQi_YDb_HEUF8VsXvgmdnDjEY/export?format=csv&gid=1730452293"

    # Define the data types for each column (excluding Duration for now)
    dtype_dict = {
        "Distance": "float",
        "Price": "float",
    }

    # Read the CSV into a DataFrame with specified data types
    raw_df = pd.read_csv(sheet_url, dtype=dtype_dict, parse_dates=["Timestamp"])
    df = raw_df.copy()
    raw_df = raw_df.drop(columns=["Timestamp", "Comment (optional)"])

    # Convert the Duration column to timedelta
    df['Duration'] = pd.to_timedelta(df['Duration'])

    # Create a column with the duration in minutes
    df['Duration_minutes'] = df['Duration'].dt.total_seconds() / 60

    # Remove outliers
    df = df[df['Price'] > 3]
    df = df[df['Price'] < 80]
    df = df[df['Distance'] < 25]
    df = df[df['Duration_minutes'] < 90]

    # Perform multiple linear regression
    X1 = df[['Distance', 'Duration_minutes']]
    y1 = df['Price']
    model = LinearRegression()
    model.fit(X1, y1)
    r2 = model.score(X1, y1)

    # Create a model for each city
    area_list = df['Area'].unique()
    city_models = {}
    city_r2 = {}
    for area in area_list:
        X_area = df[df['Area'] == area][['Distance', 'Duration_minutes']]
        y_area = df[df['Area'] == area]['Price']
        model_area = LinearRegression()
        model_area.fit(X_area, y_area)
        city_models[area] = model_area
        city_r2[area] = model_area.score(X_area, y_area)


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
    st.write(f"Price = \\${model.intercept_:.2f} + \\${model.coef_[0]:.2f} per mile + \\${model.coef_[1]:.2f} per minute (R² = {r2:.2f})")
    st.subheader(f"Area models coefficients")
    reg_string = ""
    for area in area_list:
        reg_string += f"- {area}: Price = \\${city_models[area].intercept_:.2f} + \\${city_models[area].coef_[0]:.2f} per mile + \\${city_models[area].coef_[1]:.2f} per minute (R² = {city_r2[area]:.2f})\n"
    st.write(reg_string)
    # st.write(f"_San Francisco and Phoenix model coefficients are coming when more data is available_")

    # Scatter plots for distance vs. price and duration vs. price
    st.subheader("Scatter plots")

    col1, col2 = st.columns(2)

    with col1:
        st.scatter_chart(df, x='Distance', y='Price', color='Area')

    with col2:
        st.scatter_chart(df, x='Duration_minutes', y='Price', color='Area')

    st.subheader("24h Price Fluctuation")
    all_city_data = pd.DataFrame()

    for city in df['Area'].unique():
        df_city = df[df['Area'] == city]
        df_city['Price per Distance'] = df_city['Price'] / df_city['Distance']
        df_city['Hour'] = df_city['Timestamp'].dt.hour
        df_city['Area'] = city
        all_city_data = pd.concat([all_city_data, df_city], ignore_index=True)

    # Remove outliers
    all_city_data_filtered = all_city_data[(np.abs(stats.zscore(all_city_data['Price per Distance'])) < 3)]

    # Extend data to include hours above and below
    data_22_to_24 = all_city_data_filtered[all_city_data_filtered['Hour'] >= 22].copy()
    data_22_to_24['Hour'] = data_22_to_24['Hour'] - 24
    data_0_to_2 = all_city_data_filtered[all_city_data_filtered['Hour'] <= 2].copy()
    data_0_to_2['Hour'] = data_0_to_2['Hour'] + 24
    all_city_data_extended = pd.concat([data_22_to_24, all_city_data_filtered, data_0_to_2], ignore_index=True)

    # Polynomial fit order
    poly_fit_order = 7
    coefficients = np.polyfit(all_city_data_extended['Hour'], all_city_data_extended['Price per Distance'], poly_fit_order)
    polynomial = np.poly1d(coefficients)

    x_curve = np.linspace(0, 24, 100)
    y_curve = polynomial(x_curve)

    # Calculate 95% confidence interval
    x2, y2, lower, upper = prediction_interval(all_city_data_extended['Hour'], all_city_data_extended['Price per Distance'], poly_fit_order, 0, 24)

    curve_data = pd.DataFrame({
        "Hour": x_curve,
        "Price per Distance": y_curve,
        "Type": "Fitted Curve"
    })
    prediction_interval_data = pd.DataFrame({
        "Hour": x2,
        "Lower Bound": lower,
        "Upper Bound": upper
    })

    prediction_interval_curve = alt.Chart(prediction_interval_data).mark_area(opacity=0.3).encode(
        x='Hour',
        y='Lower Bound',
        y2='Upper Bound',
        color=alt.value('lightgrey')
    )
    scatter = alt.Chart(all_city_data).mark_point(filled=True).encode(
        x=alt.X('Hour:Q', title='Hour'),
        y=alt.Y('Price per Distance:Q', title='Price per Distance'),
        color=alt.Color('Area:N', legend=alt.Legend(orient='bottom'))
    )
    line = alt.Chart(curve_data).mark_line().encode(
        x='Hour',
        y='Price per Distance',
        color=alt.value('green')
    )

    # Combine the charts
    chart = scatter + line + prediction_interval_curve
    chart = chart.configure_axisX(grid=False) 

    st.altair_chart(chart, use_container_width=True)

    # Input form for price prediction
    st.subheader("Predict Price")
    distance_input = st.slider("Select distance in miles:", min_value=0.0, max_value=20.0, value=6.0, step=0.25)
    duration_input = st.slider("Select duration in minutes:", min_value=0, max_value=90, value=30, step=1)


    city_input = st.selectbox("Select area:", options=["Los Angeles", "San Francisco", "Phoenix"])

    gen_prediction = model.predict([[distance_input, duration_input]])[0]
    city_prediction = city_models[city_input].predict([[distance_input, duration_input]])[0]

    st.markdown(f"Predicted general price: \\$ **{gen_prediction:.2f}**")
    st.markdown(f"Predicted price for {city_input}: \\$ **{city_prediction:.2f}**")

    # Histograms for Travel Distance, Travel Time, and Price
    st.subheader("Travel distance, travel time, and price histograms")

    col1, col2, col3 = st.columns(3)

    # Binning the data and converting to string for better label display
    distances = df['Distance'].sort_values().value_counts(bins=range(0, 19, 1), sort=False).to_frame()
    durations = df['Duration_minutes'].sort_values().value_counts(bins=range(0, 65, 5), sort=False).to_frame()
    prices = df['Price'].sort_values().value_counts(bins=range(0, 42, 2), sort=False).to_frame()

    distances.index = [f"{int(i.left):02d}-{int(i.right):02d}" for i in distances.index]
    durations.index = [f"{int(i.left):02d}-{int(i.right):02d}" for i in durations.index]
    prices.index = [f"$ {int(i.left):02d}-{int(i.right):02d}" for i in prices.index]

    with col1:
        st.bar_chart(distances, x_label='Distance (miles)')

    with col2:
        st.bar_chart(durations, x_label='Duration (minutes)')

    with col3:
        st.bar_chart(prices, x_label='Price ($)')

    # Add two histograms: Price per day of the week and Price per hour of the day
    # Extract day of the week and hour of the day from the Timestamp
    df['Hour_of_day'] = df['Timestamp'].dt.hour

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Day of week'] = pd.Categorical(df['Day of week'], categories=days_order, ordered=True)

    st.subheader("Price distributions")
    st.markdown("_Note that these might be skewed by people taking shorter or longer rides at different times._")
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df.groupby('Day of week')['Price'].mean().sort_index())
        # Price per area
        st.bar_chart(df.groupby('Area')['Price'].mean())

    with col2:
        st.bar_chart(df.groupby('Hour_of_day')['Price'].mean())
        # Price per day part
        df['DayPart'] = pd.cut(df['Hour_of_day'], bins=[0, 7, 10, 16, 19, 24], labels=['Night (0-7)', 'Morning (7-10)', 'Day (10-16)', 'Evening (16-19)', 'Night (19-24)'])
        st.bar_chart(df.groupby('DayPart')['Price'].mean())

    # Price per mile distribution
    df['Price_per_mile'] = df['Price'] / df['Distance']

    st.subheader("Price per mile distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df.groupby('Day of week')['Price_per_mile'].mean().sort_index())
        # Price per mile by area
        st.bar_chart(df.groupby('Area')['Price_per_mile'].mean())

    with col2:
        st.bar_chart(df.groupby('Hour_of_day')['Price_per_mile'].mean())
        # Price per mile by day part
        st.bar_chart(df.groupby('DayPart')['Price_per_mile'].mean())

    # Price per minute distribution
    df['Price_per_minute'] = df['Price'] / df['Duration_minutes']

    st.subheader("Price per minute distributions")
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df.groupby('Day of week')['Price_per_minute'].mean().sort_index())
        # Price per minute by area
        st.bar_chart(df.groupby('Area')['Price_per_minute'].mean())

    with col2:
        st.bar_chart(df.groupby('Hour_of_day')['Price_per_minute'].mean())
        # Price per minute by day part
        st.bar_chart(df.groupby('DayPart')['Price_per_minute'].mean())


if __name__ == "__main__":
    main()
