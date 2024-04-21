# Data Source: https://public.tableau.com/app/profile/federal.trade.commission/viz/FraudandIDTheftMaps/AllReportsbyState
# US State Boundaries: https://public.opendatasoft.com/explore/dataset/us-state-boundaries/export/

import streamlit as st  
import altair as alt
import numpy as np  
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

PAGE_TITLE = 'HOME VALUE APPRECIATION'
PAGE_SUB_TITLE = "Home value appreciation refers to the increase in the value of a property over time. \
The primary benefit of appreciation is the potential for capital gains. It affects both the long-term profitability \
and the strategic approach to property investment."
hpa_formula = r"HPA = \frac{\text{Home value}_t - \text{Home value}_0}{\text{Home value}_0}"


def df_time_filter(df, start_date, end_date, data_type):
    if data_type == 'wide':
        date_columns = pd.to_datetime(df.columns, errors='coerce', format="%Y-%m-%d")
        # Filter columns that were successfully converted to datetime
        is_date_column = np.array([isinstance(col, pd.Timestamp) for col in date_columns ])
        
        subset_columns = df.columns[~is_date_column].union(df.columns[(date_columns >= start_date) & (date_columns <= end_date)])
        subset_df = df[subset_columns]
        date_columns_ls = df.columns[(date_columns >= start_date) & (date_columns <= end_date)]
        return subset_df, date_columns_ls

    elif data_type == 'long':
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        subset_df = df[(df['Date']>=start_date)&(df['Date']<=end_date)]
        return subset_df
    

def get_cities_data(df, date_columns):
    '''
      calculate the average of the given time range
    '''

    cities_ts_data = df[date_columns]

    cities_data = df[['RegionID', 'RegionName', 'StateName', 'population', 'density', 'lng', 'lat']]
    cities_data['HomeValue'] = cities_ts_data.mean(axis=1)
    cities_data['HomeValueAppreciation'] = (df[date_columns[-1]] - df[date_columns[0]]) / df[date_columns[0]] *100

    cities_data.dropna(inplace=True)

    # Create Point geometries from latitude and longitude
    geometry = [Point(xy) for xy in zip(cities_data['lng'], cities_data['lat'])]
    # Convert DataFrame to GeoDataFrame
    cities_geoPandas = gpd.GeoDataFrame(cities_data, geometry=geometry)
    # Set the CRS for the GeoDataFrame
    cities_geoPandas.crs = 'EPSG:4326'  # Assuming WGS84 coordinate reference system
    # Drop the latitude and longitude columns if needed
    cities_geoPandas = cities_geoPandas.drop(['lat', 'lng'], axis=1)
    #cities_geoPandas = cities_geoPandas.rename(columns={date:'HomeValue'})
    return cities_geoPandas

def get_states_geoJson():
    states_geoJson = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
    ).json()
    return states_geoJson
    
def get_state_level_data(df, date_columns):

    df['HomeValue'] = df[date_columns].mean(axis=1)
    df['HomeValueAppreciation'] = (df[date_columns[-1]] - df[date_columns[0]]) / df[date_columns[0]] *100

    state_level_data = df[['StateName', 'HomeValue', 'HomeValueAppreciation', 'population', 'density']]
    return state_level_data
    
def plot_map(states_geoJson, state_level_data, cities_geoPandas):
    m = folium.Map(location=[38, -102], zoom_start=4, scrollWheelZoom=False)

    # add color layer to the map
    choropleth = folium.Choropleth(
        geo_data=states_geoJson,
        name="choropleth",
        data=state_level_data,
        columns=["StateName", "HomeValueAppreciation"],
        key_on="feature.id",
        fill_color="Greens", #BuPu
        bins=9,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Home Value Appreciation(%)",
        highlight=True,
    ).add_to(m)

    # add tooltip when hover the mouse over
    # add HomeValue into geojson data in order to show it
    for feature in choropleth.geojson.data['features']:
        state_name = feature['id']
        HomeValue = state_level_data.loc[state_level_data['StateName']==state_name, 'HomeValue'].values[0]
        HomeValueAppreciation = state_level_data.loc[state_level_data['StateName']==state_name, 'HomeValueAppreciation'].values[0]
        feature['properties']['HomeValue'] = f'State Avg Home Value: ${HomeValue:.2f}'
        feature['properties']['HomeValueAppreciation'] = f'State Home Value Appreciation: {HomeValueAppreciation:.2f}%'
        

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name', 'HomeValueAppreciation', 'HomeValue'], labels=False)
    )
    
    
    # add cities
    folium.GeoJson(
        cities_geoPandas,
        name="Subway Stations",
        marker=folium.Circle(radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1),
        tooltip=folium.GeoJsonTooltip(fields=["RegionName", 'HomeValueAppreciation', 'HomeValue', 'population', 'density']), # 
        popup=folium.GeoJsonPopup(fields=["RegionName", 'HomeValueAppreciation', 'HomeValue', 'population', 'density',]),  # 
        style_function=lambda x: {
            "radius": (x['properties']['HomeValue'])*0.08,
        },
        highlight_function=lambda x: {"fillOpacity": 0.8},
        zoom_on_click=False,
    ).add_to(m)
    
    # Add dark and light mode. 
    folium.TileLayer('cartodbdark_matter',name="dark mode",control=True).add_to(m)
    folium.TileLayer('cartodbpositron',name="light mode",control=True).add_to(m)

    # We add a layer controller. 
    folium.LayerControl(collapsed=True).add_to(m)
    st_map = st_folium(m, width=700, height=450)
    
    state = 'USA'
    state_name = 'USA'
    city = ''
    if st_map['last_active_drawing']:
        try:
            city = st_map['last_active_drawing']['properties']['RegionName']
            state = st_map['last_active_drawing']['properties']['StateName'] 
        except:
            # It's State
            # state_id = st_map['last_active_drawing']
            state = st_map['last_active_drawing']['id']
            state_name = st_map['last_active_drawing']['properties']['name']
    #st.write(st_map)
    return state, state_name, city

def display_state_filter(df, state_name):
    state_list = ['USA'] + sorted(df.state_name.unique().tolist())
    state_index = state_list.index(state_name)
    return st.sidebar.selectbox("State", state_list, state_index)
    

def display_date_filter(df):
    # filter the time range
    col1, col2, col3 = st.columns([10, 10, 4])
    date_list = df.Date.tolist()[::-1]
    start_date = col1.selectbox("Start Date", date_list, len(date_list)-1)
    end_date = col2.selectbox("End Date", date_list, 0)
    return start_date, end_date



def line_chart(df, subset_df_newc_count_US, metric_name, yaxis_label):
    # time series line chart
    ## reference flat line at a specific newc_count value
    ## Create a DataFrame for the reference line, reference period '2018-4-30 - 2020-4-30'
    reference_df = df[(df['Date'] >= '2018-04-30')&(df['Date'] <= '2020-04-30')]
    reference_value_avg = reference_df[metric_name].mean()
    reference_value_sd = np.std(reference_df[metric_name])
    reference_line_df = pd.DataFrame({
        'Date': [df['Date'].min(), df['Date'].max()],
        metric_name: [reference_value_avg, reference_value_avg],
        f'{metric_name}_plus_sd': [reference_value_avg+reference_value_sd*2, reference_value_avg+reference_value_sd*2],
        f'{metric_name}_minus_sd': [reference_value_avg-reference_value_sd*2, reference_value_avg-reference_value_sd*2]  
    })
    reference_line = alt.Chart(reference_line_df).mark_line(
        color='purple', strokeDash=[5, 5], size=2
    ).encode(
        x='yearmonth(Date):T',
        y=f'{metric_name}:Q'
    )
    reference_line_plus_sd = alt.Chart(reference_line_df).mark_line(
        color='green', strokeDash=[5, 5], size=2
    ).encode(
        x='yearmonth(Date):T',
        y=f'{metric_name}_plus_sd:Q'
    )
    reference_line_minus_sd = alt.Chart(reference_line_df).mark_line(
        color='red', strokeDash=[5, 5], size=2
    ).encode(
        x='yearmonth(Date):T',
        y=f'{metric_name}_minus_sd:Q'
    )


    # Create an Altair chart object
    line = alt.Chart(df).mark_line(
        color='steelblue',  # Customize line color
        size=3  # Customize line width
    ).encode(
        x=alt.X('yearmonth(Date):T', title='', axis=alt.Axis(format='%Y-%m')),  # Specify the data type (T for temporal)
        y=alt.Y(f'{metric_name}:Q', title=yaxis_label),  # Custom axis title
        tooltip=['Date', metric_name]  # Tooltip for interactivity
    )
    
    # Create points for each data entry
    points = alt.Chart(df).mark_point(
        color='steelblue',  # Match line color or choose a different one for contrast
        size=50,  # Adjust size of the point markers
    ).encode(
        x=alt.X('yearmonth(Date):T', title=''),
        y=alt.Y(f'{metric_name}:Q', title=yaxis_label),
        tooltip=['Date', metric_name]  # Tooltip for interactivity
    )

    # Combine the line and points
    chart = (line + points + reference_line + reference_line_plus_sd + reference_line_minus_sd).properties(
        width='container',  # Use container width
        height=250,  # Custom height
        #title=state + ' Monthly New Construction Count Over Time'  # Chart title
    ).configure_view(
        strokeWidth=0  # Remove border around chart
    ).configure_axis(
        gridColor='lightgray'  # Customize grid color
    )
    return chart

def convert_date_format(col_name):
    try:
        # Try to convert the column name to a date
        dt = datetime.strptime(col_name, '%m/%d/%Y')
        # Return the reformatted date string
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        # If an error occurs, it's not a date, return the original column name
        return col_name

def bar_chart(data, xmetric_name, ymetric_name, color_name, order, xaxis_label, yaxis_label, chart_title):
    chart = alt.Chart(data).mark_bar(color=color_name).encode(
    y=alt.Y(f'{ymetric_name}:N', sort=alt.EncodingSortField(field=xmetric_name, order=order),  title=yaxis_label),  # 'N' indicates nominal (categorical) data
    x=alt.X(f'{xmetric_name}:Q', sort='-x', title=xaxis_label),  # 'Q' indicates quantitative data
    #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
    tooltip=[ymetric_name, xmetric_name]  # Show tooltip on hover
    ).properties(
    title=chart_title, 
    width='container',  # Make the chart responsive
    height=400  # Set a fixed height for the chart
    )
    return chart

def main():
    st.set_page_config(
     page_title=PAGE_TITLE,
     layout="wide",
     initial_sidebar_state="expanded",
    )
    st.title(PAGE_TITLE)
    st.caption(PAGE_SUB_TITLE)
    st.caption(f"Home Price Appreciation (HPA) formula: ${hpa_formula}$")
    st.markdown("All the information provided below are within the time range selected here:")

    #Load Data
    mortgage_rate = pd.read_csv('../data/home_value/MORTGAGE30US.csv')
    df_HomeValue_cities = pd.read_csv('../data/home_value/yw1_states_data.csv')
    df_HomeValue_states = pd.read_csv('../data/home_value/yw1_us_state_data.csv')
    df_HomeValue_US = pd.read_csv('../data/home_value/yw1_us_data.csv')

    df_HomeValue_cities.columns = [convert_date_format(col) for col in df_HomeValue_cities.columns]
    df_HomeValue_states.columns = [convert_date_format(col) for col in df_HomeValue_states.columns]
    df_HomeValue_US['Date'] = df_HomeValue_US['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').strftime("%Y-%m-%d"))

    start_date, end_date = display_date_filter(df_HomeValue_US)
    if end_date < start_date:
        st.error('Date Range Error: End Date Should be Later than the Start Date')
    #state = display_state_filter(df_HomeValue_states, 'StateName')
    subset_df_HomeValue_cities, date_columns_ls = df_time_filter(df_HomeValue_cities, start_date, end_date, 'wide')
    subset_df_HomeValue_states, _ = df_time_filter(df_HomeValue_states, start_date, end_date, 'wide')
    subset_df_HomeValue_US = df_time_filter(df_HomeValue_US, start_date, end_date, 'long')

    cities_geoPandas = get_cities_data(subset_df_HomeValue_cities, date_columns_ls)
    states_geoJson = get_states_geoJson()
    state_level_data = get_state_level_data(subset_df_HomeValue_states, date_columns_ls)


    # Create two columns with width ratios 3:1
    col1, col2, col3 = st.columns([24, 1, 7])
    # Add content to the first column (3/4 page width)
    with col1:
        st.subheader("Home Value Appreciation Across Regions")
        st.caption("Deeper color represents higher ratio for States, \nlarger circle size represents higher ratio for MSAs")
        # Add your content here for the main column
        state, state_name, city = plot_map(states_geoJson, state_level_data, cities_geoPandas)
        if city:
            state = subset_df_HomeValue_states.loc[subset_df_HomeValue_states['StateName']==state]['StateName'].values[0]

    with col3:
        st.subheader(" ")
        # USA
        if state == 'USA':
            US_HomeValue = subset_df_HomeValue_US['HomeValue'].mean()
            US_HomeValueAppreciation = (subset_df_HomeValue_US['HomeValue'].iloc[-1] - subset_df_HomeValue_US['HomeValue'].iloc[0]) / subset_df_HomeValue_US['HomeValue'].iloc[0] *100
            st.metric(label="COUNTRY", value="USA")
            st.metric(label="Home Value Appreciation", value=f"{US_HomeValueAppreciation:,.2f}%")
            st.metric(label="AVG Home Value", value=f"${US_HomeValue:,.2f}")

        else:
            if city == '':
                # State
                #df = subset_df_HomeValue_states[(subset_df_HomeValue_states['StateName']==state)]
                state_data = state_level_data[state_level_data['StateName']==state]
                
                st.metric(label="STATE", value=state_name)
                st.metric(label="Home Value Appreciation", value=f"{state_data['HomeValueAppreciation'].values[0]:,.2f}%")
                st.metric(label="AVG Home Value", value=f"${state_data['HomeValue'].values[0]:,.2f}")
                #st.metric(label="POPULATION", value=f"{state_data['Population'].values[0]:,.2f}")
                #st.metric(label="DENSITY / Km2", value=f"{state_data['Density'].values[0]:,.2f}")
            else:
                # city
                city_data = cities_geoPandas.loc[(cities_geoPandas['RegionName']==city) &\
                                                 (cities_geoPandas['StateName']==state)]

                st.metric(label="CITY", value=city)
                st.metric(label="Home Value Appreciation", value=f"{city_data['HomeValueAppreciation'].values[0]:,.2f}%")
                st.metric(label="AVG Home Value", value=f"${city_data['HomeValue'].values[0]:,.2f}")
                #st.metric(label="POPULATION", value=f"{city_data['population'].values[0]:,.2f}")
                #st.metric(label="DENSITY / Km2", value=f"{city_data['density'].values[0]:,.2f}")

    #Display Filters and Map


    # top 10 regions bar chart - by home value appreciation
    col1, col2, col3 = st.columns([15, 2, 15])
    if state == 'USA':
        # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
        df_sorted_head = state_level_data.sort_values(by='HomeValueAppreciation', ascending=False).head(10)
        df_sorted_tail = state_level_data.sort_values(by='HomeValueAppreciation', ascending=True).head(10)

        with col1:
            # Create an Altair bar chart
            chart = bar_chart(data=df_sorted_head, xmetric_name='HomeValueAppreciation', ymetric_name='StateName', color_name='#2ca25f', order='descending',
                xaxis_label='Home Value Appreciation', yaxis_label='State', chart_title='Top 10 States by Home Value Appreciation')
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

        with col3:
            # Create an Altair bar chart
            chart = bar_chart(data=df_sorted_tail, xmetric_name='HomeValueAppreciation', ymetric_name='StateName', color_name='#e34a33', order='ascending',
                xaxis_label='Home Value Appreciation', yaxis_label='State', chart_title='The Worst 10 States by Home Value Appreciation')
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

    else:
        if city == '':
            # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
            cities_data_one_state = cities_geoPandas[cities_geoPandas['StateName']==state].drop(columns=['geometry'])
            df_sorted_head = cities_data_one_state.sort_values(by='HomeValueAppreciation', ascending=False).head(10)
            df_sorted_tail = cities_data_one_state.sort_values(by='HomeValueAppreciation', ascending=True).head(10)

            with col1:
                # Create an Altair bar chart
                chart = bar_chart(data=df_sorted_head, xmetric_name='HomeValueAppreciation', ymetric_name='RegionName', 
                                  color_name='#2ca25f', order='descending',
                              xaxis_label='Home Value Appreciation', yaxis_label='MSA', chart_title='Top 10 MSAs by Home Value Appreciation')
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)

            with col3:
                # Create an Altair bar chart
                chart = bar_chart(data=df_sorted_tail, xmetric_name='HomeValueAppreciation', ymetric_name='RegionName', 
                                  color_name='#e34a33', order='ascending',
                              xaxis_label='Home Value Appreciation', yaxis_label='MSA', chart_title='The Worst 10 MSAs by Home Value Appreciation')
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)

    # top 10 regions bar chart - by home value
    col1, col2, col3 = st.columns([15, 2, 15])
    if state == 'USA':
        # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
        df_sorted_head = state_level_data.sort_values(by='HomeValue', ascending=False).head(10)
        df_sorted_tail = state_level_data.sort_values(by='HomeValue', ascending=True).head(10)

        with col1:
            # Create an Altair bar chart
            chart = bar_chart(data=df_sorted_head, xmetric_name='HomeValue', ymetric_name='StateName', color_name='#99d8c9', order='descending',
                xaxis_label='Absolute Home Value', yaxis_label='State', chart_title='Top 10 States by Absolute Home Value')
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

        with col3:
            # Create an Altair bar chart
            chart = bar_chart(data=df_sorted_tail, xmetric_name='HomeValue', ymetric_name='StateName', color_name='#fc9272', order='ascending',
                xaxis_label='Absolute Home Value', yaxis_label='State', chart_title='The Worst 10 States by Absolute Home Value')
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

    else:
        if city == '':
            # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
            cities_data_one_state = cities_geoPandas[cities_geoPandas['StateName']==state].drop(columns=['geometry'])
            df_sorted_head = cities_data_one_state.sort_values(by='HomeValue', ascending=False).head(10)
            df_sorted_tail = cities_data_one_state.sort_values(by='HomeValue', ascending=True).head(10)

            with col1:
                # Create an Altair bar chart
                chart = bar_chart(data=df_sorted_head, xmetric_name='HomeValue', ymetric_name='RegionName', 
                                  color_name='#99d8c9', order='descending',
                              xaxis_label='Absolute Home Value', yaxis_label='MSA', chart_title='Top 10 MSAs by Absolute Home Value')
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)

            with col3:
                # Create an Altair bar chart
                chart = bar_chart(data=df_sorted_tail, xmetric_name='HomeValue', ymetric_name='RegionName', 
                                  color_name='#fc9272', order='ascending',
                              xaxis_label='Absolute Home Value', yaxis_label='MSA', chart_title='The Worst 10 MSAs by Absolute Home Value')
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)



    # time series line chart - home value
    if state == 'USA':
        st.subheader(state + ' Monthly Home Value Over Time')
        st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
        chart = line_chart(subset_df_HomeValue_US, subset_df_HomeValue_US, 'HomeValue', 'Home Value')
        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)
    
    else:
        if city == '':
            st.subheader(state_name + ' Monthly Home Value Over Time')
            st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
            df = subset_df_HomeValue_states[(subset_df_HomeValue_states['StateName']==state)][date_columns_ls].mean(axis=0)
            df = df.reset_index(drop=False).rename(columns={'index':'Date', 0: 'HomeValue'})
            
        else:
            st.subheader(city + ' Monthly Home Value Over Time')
            st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
            df = subset_df_HomeValue_cities.loc[(subset_df_HomeValue_cities['RegionName']==city) &\
                                            (subset_df_HomeValue_cities['StateName']==state)][date_columns_ls]
            df = df.T.reset_index(drop=False)
            df.columns = ['Date', 'HomeValue']       
        #st.line_chart(data = df, x='Date', y='HomeValue', height=250, use_container_width=True)
        chart = line_chart(df, subset_df_HomeValue_US, 'HomeValue', 'Home Value')
        st.altair_chart(chart, use_container_width=True)

    # tims series line chart - 30yr mortgage rate
    mortgage_rate['DATE'] = pd.to_datetime(mortgage_rate['DATE'], format='%Y-%m-%d')
    subset_mortgage_rate = mortgage_rate[(mortgage_rate['DATE'] >= start_date) & (mortgage_rate['DATE'] <= end_date)]
    st.subheader('US Monthly Mortagage Rate(Fixed 30 Years) Over Time')
    # Create an Altair chart object
    line = alt.Chart(subset_mortgage_rate).mark_line(
        color='orange',  # Customize line color
        size=3  # Customize line width
    ).encode(
        x=alt.X('yearmonth(DATE):T', title='', axis=alt.Axis(format='%Y-%m')),  # Specify the data type (T for temporal)
        y=alt.Y('MORTGAGE30US:Q', title='Mortgage Rate F30yr'),  # Custom axis title
        tooltip=['DATE', 'MORTGAGE30US']  # Tooltip for interactivity
    )
    
    # Create points for each data entry
    points = alt.Chart(subset_mortgage_rate).mark_point(
        color='orange',  # Match line color or choose a different one for contrast
        size=50,  # Adjust size of the point markers
    ).encode(
        x=alt.X('yearmonth(DATE):T', title=''),
        y=alt.Y('MORTGAGE30US:Q', title='Mortgage Rate F30yr'),
        tooltip=['DATE', 'MORTGAGE30US']  # Tooltip for interactivity
    )

    # Combine the line and points
    chart = (line + points ).properties(
        width='container',  # Use container width
        height=250,  # Custom height
        #title=state + ' Monthly New Construction Count Over Time'  # Chart title
    ).configure_view(
        strokeWidth=0  # Remove border around chart
    ).configure_axis(
        gridColor='lightgray'  # Customize grid color
    )
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    

    main()
