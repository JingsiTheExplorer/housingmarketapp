# Data Source: https://public.tableau.com/app/profile/federal.trade.commission/viz/FraudandIDTheftMaps/AllReportsbyState
# US State Boundaries: https://public.opendatasoft.com/explore/dataset/us-state-boundaries/export/

import streamlit as st  
import altair as alt
import numpy as np  
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

PAGE_TITLE = 'SUPPLY AND DEMAND'
PAGE_SUB_TITLE = 'New listings and new pendings reflect the supply and demand situation of the real estate market to a certain extent. Here, we use (new pendings count / (inventory + new listing count)) ratios to compare the situations in different areas. The higher the ratio, the less effort to sell.'

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
        df['Date'] = pd.to_datetime(df['Date'])
        subset_df = df[(df['Date']>=start_date)&(df['Date']<=end_date)]
        #st.write(subset_df)
        return subset_df
    

def get_cities_data(df, date_columns):
    '''
      calculate the average of the given time range
    '''

    cities_ts_data = df[date_columns]

    cities_data = df[['RegionID', 'RegionName', 'StateName', 'lng', 'lat']]
    cities_data['demand_supply_ratio'] = cities_ts_data.mean(axis=1)

    cities_data.dropna(inplace=True)

    # Create Point geometries from latitude and longitude
    geometry = [Point(xy) for xy in zip(cities_data['lng'], cities_data['lat'])]
    # Convert DataFrame to GeoDataFrame
    cities_geoPandas = gpd.GeoDataFrame(cities_data, geometry=geometry)
    # Set the CRS for the GeoDataFrame
    cities_geoPandas.crs = 'EPSG:4326'  # Assuming WGS84 coordinate reference system
    # Drop the latitude and longitude columns if needed
    cities_geoPandas = cities_geoPandas.drop(['lat', 'lng'], axis=1)
    #cities_geoPandas = cities_geoPandas.rename(columns={date:'demand_supply_ratio'})
    return cities_geoPandas

def get_states_geoJson():
    states_geoJson = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
    ).json()
    return states_geoJson
    
def get_state_level_data(df, date_columns):

    df['metric_mean_over_time_bystate'] = df[date_columns].mean(axis=1)

    state_level_data = df.groupby(['StateName']) \
            .apply(lambda x: pd.Series({
                'demand_supply_ratio': x['metric_mean_over_time_bystate'].mean(),
                #'Population': x['population'].mean(),
                #'Density': x['density'].mean(),
                })).reset_index()

    return state_level_data
    
def plot_map(states_geoJson, state_level_data, cities_geoPandas):
    m = folium.Map(location=[38, -102], zoom_start=4, scrollWheelZoom=False)

    # add color layer to the map
    choropleth = folium.Choropleth(
        geo_data=states_geoJson,
        name="choropleth",
        data=state_level_data,
        columns=["StateName", "demand_supply_ratio"],
        key_on="feature.id",
        fill_color="PuBuGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="demand_supply_ratio(%)",
        highlight=True,
    ).add_to(m)

    # add tooltip when hover the mouse over
    # add demand_supply_ratio into geojson data in order to show it
    for feature in choropleth.geojson.data['features']:
        state_name = feature['id']
        demand_supply_ratio = state_level_data.loc[state_level_data['StateName']==state_name, 'demand_supply_ratio'].values[0]
        feature['properties']['demand_supply_ratio'] = f'State Avg demand_supply_ratio: {demand_supply_ratio:.2f}'
        

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name', 'demand_supply_ratio'], labels=False)
    )
    
    # add cities
    folium.GeoJson(
        cities_geoPandas,
        name="Subway Stations",
        marker=folium.Circle(radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1),
        tooltip=folium.GeoJsonTooltip(fields=["RegionName",  'demand_supply_ratio']), # 'population', 'density',
        popup=folium.GeoJsonPopup(fields=["RegionName", 'demand_supply_ratio']),  # 'population', 'density', 
        style_function=lambda x: {
            "radius": (x['properties']['demand_supply_ratio'])*80000,
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



def line_chart(df, subset_df_demand_supply_ratio_US):
    # time series line chart
    ## reference flat line at a specific demand_supply_ratio value
    ## Create a DataFrame for the reference line, reference period '2018-4-30 - 2020-4-30'
    reference_df = subset_df_demand_supply_ratio_US[(subset_df_demand_supply_ratio_US['Date'] >= '2018-04-30')&(subset_df_demand_supply_ratio_US['Date'] <= '2020-04-30')]
    reference_value_avg = reference_df['demand_supply_ratio'].mean()
    reference_value_sd = np.std(reference_df['demand_supply_ratio'])
    reference_line_df = pd.DataFrame({
        'Date': [subset_df_demand_supply_ratio_US['Date'].min(), subset_df_demand_supply_ratio_US['Date'].max()],
        'demand_supply_ratio': [reference_value_avg, reference_value_avg],
        'demand_supply_ratio_plus_sd': [reference_value_avg+reference_value_sd*2, reference_value_avg+reference_value_sd*2],
        'demand_supply_ratio_minus_sd': [reference_value_avg-reference_value_sd*2, reference_value_avg-reference_value_sd*2]  
    })
    reference_line = alt.Chart(reference_line_df).mark_line(
        color='purple', strokeDash=[5, 5], size=2
    ).encode(
        x='yearmonth(Date):T',
        y='demand_supply_ratio:Q'
    )
    reference_line_plus_sd = alt.Chart(reference_line_df).mark_line(
        color='green', strokeDash=[5, 5], size=2
    ).encode(
        x='yearmonth(Date):T',
        y='demand_supply_ratio_plus_sd:Q'
    )
    reference_line_minus_sd = alt.Chart(reference_line_df).mark_line(
        color='red', strokeDash=[5, 5], size=2
    ).encode(
        x='yearmonth(Date):T',
        y='demand_supply_ratio_minus_sd:Q'
    )


    #st.line_chart(data = subset_df_demand_supply_ratio_US, x='Date', y='demand_supply_ratio', height=250, use_container_width=True)
    # Create an Altair chart object
    line = alt.Chart(df).mark_line(
        color='steelblue',  # Customize line color
        size=3  # Customize line width
    ).encode(
        x=alt.X('yearmonth(Date):T', title='', axis=alt.Axis(format='%Y-%m')),  # Specify the data type (T for temporal)
        y=alt.Y('demand_supply_ratio:Q', title='Demand Supply Ratio'),  # Custom axis title
        tooltip=['Date', 'demand_supply_ratio']  # Tooltip for interactivity
    )
    
    # Create points for each data entry
    points = alt.Chart(df).mark_point(
        color='steelblue',  # Match line color or choose a different one for contrast
        size=50,  # Adjust size of the point markers
    ).encode(
        x=alt.X('yearmonth(Date):T', title=''),
        y=alt.Y('demand_supply_ratio:Q', title='Demand Supply Ratio'),
        tooltip=['Date', 'demand_supply_ratio']  # Tooltip for interactivity
    )

    # Combine the line and points
    chart = (line + points + reference_line + reference_line_plus_sd + reference_line_minus_sd).properties(
        width='container',  # Use container width
        height=250,  # Custom height
        #title=state + ' Monthly Demand/Supply Ratio Over Time'  # Chart title
    ).configure_view(
        strokeWidth=0  # Remove border around chart
    ).configure_axis(
        gridColor='lightgray'  # Customize grid color
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
    st.markdown("All the information provided below are within the time range selected here:")

    #Load Data
    df_demand_supply_ratio_states = pd.read_csv('data/new_pend_ls/demand_supply_ratio_state.csv')
    df_demand_supply_ratio_US = pd.read_csv('data/new_pend_ls/demand_supply_ratio_us.csv')
    response = requests.get('https://files.zillowstatic.com/research/public_csvs/new_listings/Metro_new_listings_uc_sfrcondo_sm_month.csv?t=1712721803')
    new_listing = pd.read_csv(StringIO(response.text))
    #st.write(new_listing.head())


    start_date, end_date = display_date_filter(df_demand_supply_ratio_US)
    if end_date < start_date:
        st.error('Date Range Error: End Date Should be Later than the Start Date')
    #state = display_state_filter(df_demand_supply_ratio_states, 'StateName')
    subset_df_demand_supply_ratio_states, date_columns_ls = df_time_filter(df_demand_supply_ratio_states, start_date, end_date, 'wide')
    subset_df_demand_supply_ratio_US = df_time_filter(df_demand_supply_ratio_US, start_date, end_date, 'long')

    cities_geoPandas = get_cities_data(subset_df_demand_supply_ratio_states, date_columns_ls)
    states_geoJson = get_states_geoJson()
    state_level_data = get_state_level_data(subset_df_demand_supply_ratio_states, date_columns_ls)

    


    # Create two columns with width ratios 3:1
    col1, col2, col3 = st.columns([24, 1, 7])
    # Add content to the first column (3/4 page width)
    with col1:
        st.subheader("Averaged Supply/Demand Ratio Across Regions")
        st.caption("Deeper green represents higher ratio for States, \nlarger circle size represents higher ratio for MSAs")
        # Add your content here for the main column
        state, state_name, city = plot_map(states_geoJson, state_level_data, cities_geoPandas)
        if city:
            state = subset_df_demand_supply_ratio_states.loc[subset_df_demand_supply_ratio_states['StateName']==state]['StateName'].values[0]

    with col3:
        st.subheader(" ")
        # USA
        if state == 'USA':
            US_demand_supply_ratio = subset_df_demand_supply_ratio_US['demand_supply_ratio'].mean()
            st.metric(label="COUNTRY", value="USA")
            st.metric(label="AVG SUPPLY DEMAND RATIO", value=f"{US_demand_supply_ratio:,.2f}%")
            #st.metric(label="POPULATION", value="341,814,420")
            #st.metric(label="DENCITY / Km2", value=f"{37.1:,.2f}")
        else:
            if city == '':
                # State
                #df = subset_df_demand_supply_ratio_states[(subset_df_demand_supply_ratio_states['StateName']==state)]
                state_data = state_level_data[state_level_data['StateName']==state]
                
                st.metric(label="STATE", value=state_name)
                st.metric(label="AVG SUPPLY DEMAND RATIO", value=f"{state_data['demand_supply_ratio'].values[0]:,.2f}%")
                #st.metric(label="POPULATION", value=f"{state_data['Population'].values[0]:,.2f}")
                #st.metric(label="DENCITY / Km2", value=f"{state_data['Density'].values[0]:,.2f}")
            else:
                # city
                city_data = subset_df_demand_supply_ratio_states.loc[(subset_df_demand_supply_ratio_states['RegionName']==city) &\
                                                 (subset_df_demand_supply_ratio_states['StateName']==state)]

                st.metric(label="CITY", value=city)
                st.metric(label="AVG SUPPLY DEMAND RATIO", value=f"{city_data[date_columns_ls].mean(axis=1).values[0]:,.2f}%")
                #st.metric(label="POPULATION", value=f"{city_data['population'].values[0]:,.2f}")
                #st.metric(label="DENCITY / Km2", value=f"{city_data['density'].values[0]:,.2f}")

    #Display Filters and Map

    # top 10 regions bar chart
    col1, col2, col3 = st.columns([15, 2, 15])
    if state == 'USA':
        # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
        df_sorted_head = state_level_data.sort_values(by='demand_supply_ratio', ascending=False).head(10)
        df_sorted_tail = state_level_data.sort_values(by='demand_supply_ratio', ascending=True).head(10)

        with col1:
            # Create an Altair bar chart
            chart = alt.Chart(df_sorted_head).mark_bar(color='#99d8c9').encode(
                y=alt.Y('StateName:N', sort=alt.EncodingSortField(field='demand_supply_ratio', order='descending'),  title='State'),  # 'N' indicates nominal (categorical) data
                x=alt.X('demand_supply_ratio:Q', sort='-x', title='Demand/Supply Ratio'),  # 'Q' indicates quantitative data
                #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
                tooltip=['StateName', 'demand_supply_ratio']  # Show tooltip on hover
            ).properties(
                title='Top 10 States by Ratio', 
                width='container',  # Make the chart responsive
                height=400  # Set a fixed height for the chart
            )
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

        with col3:
            # Create an Altair bar chart
            chart = alt.Chart(df_sorted_tail).mark_bar(color='#fc9272').encode(
                y=alt.Y('StateName:N', sort=alt.EncodingSortField(field='demand_supply_ratio', order='ascending'), title='State'),  # 'N' indicates nominal (categorical) data
                x=alt.X('demand_supply_ratio:Q', title='Demand/Supply Ratio'),  # 'Q' indicates quantitative data
                #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
                tooltip=['StateName', 'demand_supply_ratio']  # Show tooltip on hover
            ).properties(
                title='The Worst 10 States by Ratio', 
                width='container',  # Make the chart responsive
                height=400  # Set a fixed height for the chart
            )
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

    else:
        if city == '':
            # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
            cities_data_one_state = subset_df_demand_supply_ratio_states.loc[subset_df_demand_supply_ratio_states['StateName']==state]
            df_sorted_head = cities_data_one_state.sort_values(by='metric_mean_over_time_bystate', ascending=False).head(10)
            df_sorted_tail = cities_data_one_state.sort_values(by='metric_mean_over_time_bystate', ascending=True).head(10)

            with col1:
                # Create an Altair bar chart
                chart = alt.Chart(df_sorted_head).mark_bar(color='#99d8c9').encode(
                    y=alt.Y('RegionName:N', sort=alt.EncodingSortField(field='metric_mean_over_time_bystate', order='descending'),  title='MSA'),  # 'N' indicates nominal (categorical) data
                    x=alt.X('metric_mean_over_time_bystate:Q', sort='-x', title='Demand/Supply Ratio'),  # 'Q' indicates quantitative data
                    #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
                    tooltip=['RegionName', 'metric_mean_over_time_bystate']  # Show tooltip on hover
                ).properties(
                    title='Top 10 MSAs by Ratio', 
                    width='container',  # Make the chart responsive
                    height=400  # Set a fixed height for the chart
                )
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)

            with col3:
                # Create an Altair bar chart
                chart = alt.Chart(df_sorted_tail).mark_bar(color='#fc9272').encode(
                    y=alt.Y('RegionName:N', sort=alt.EncodingSortField(field='metric_mean_over_time_bystate', order='ascending'),  title='MSA'),  # 'N' indicates nominal (categorical) data
                    x=alt.X('metric_mean_over_time_bystate:Q', sort='-x', title='Demand/Supply Ratio'),  # 'Q' indicates quantitative data
                    #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
                    tooltip=['RegionName', 'metric_mean_over_time_bystate']  # Show tooltip on hover
                ).properties(
                    title='The Worst 10 MSAs by Ratio', 
                    width='container',  # Make the chart responsive
                    height=400  # Set a fixed height for the chart
                )
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)



    # time series line chart
    if state == 'USA':
        st.subheader(state + ' Monthly Demand/Supply Ratio Over Time')
        st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
        chart = line_chart(subset_df_demand_supply_ratio_US, subset_df_demand_supply_ratio_US)
        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)
    
    else:
        if city == '':
            st.subheader(state_name + ' Monthly Demand/Supply Ratio Over Time')
            st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
            df = subset_df_demand_supply_ratio_states[(subset_df_demand_supply_ratio_states['StateName']==state)][date_columns_ls].mean(axis=0)
            df = df.reset_index(drop=False).rename(columns={'index':'Date', 0: 'demand_supply_ratio'})
            
        else:
            st.subheader(city + ' Monthly Demand/Supply Ratio Over Time')
            st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
            df = subset_df_demand_supply_ratio_states.loc[(subset_df_demand_supply_ratio_states['RegionName']==city) &\
                                            (subset_df_demand_supply_ratio_states['StateName']==state)][date_columns_ls]
            df = df.T.reset_index(drop=False)
            df.columns = ['Date', 'demand_supply_ratio']       
        #st.line_chart(data = df, x='Date', y='demand_supply_ratio', height=250, use_container_width=True)
        chart = line_chart(df, subset_df_demand_supply_ratio_US)
        st.altair_chart(chart, use_container_width=True)



if __name__ == "__main__":
    

    main()
