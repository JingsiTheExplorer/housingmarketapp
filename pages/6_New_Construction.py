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

PAGE_TITLE = 'NEW CONSTRUCTION COUNT/PRICE'
PAGE_SUB_TITLE0 ="New construction count has impact on home value and demand."
PAGE_SUB_TITLE = 'New construction count is a smoothed measure of the typical observed market rate rent across a given region. newc_count is a repeat-rent index that is weighted to the rental housing stock to ensure representativeness across the entire market, not just those homes currently listed for-rent.'

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
    

def get_cities_data(df_count, df_price, date_columns):
    '''
      calculate the monthly metric value of the given time range
    '''

    cities_ts_data = df_count[date_columns]
    cities_ts_data_price = df_price[date_columns]

    cities_data = df_count[['RegionID', 'RegionName', 'StateName', 'lng', 'lat','population','density']]
    cities_data['newc_count'] = cities_ts_data.mean(axis=1).round(2)
    cities_data['newc_count_per_pop'] = cities_data['newc_count'] / cities_data['population']*10000
    cities_data['newc_price'] = cities_ts_data_price.mean(axis=1).round(2)

    cities_data.dropna(inplace=True)

    # Create Point geometries from latitude and longitude
    geometry = [Point(xy) for xy in zip(cities_data['lng'], cities_data['lat'])]
    # Convert DataFrame to GeoDataFrame
    cities_geoPandas = gpd.GeoDataFrame(cities_data, geometry=geometry)
    # Set the CRS for the GeoDataFrame
    cities_geoPandas.crs = 'EPSG:4326'  # Assuming WGS84 coordinate reference system
    # Drop the latitude and longitude columns if needed
    cities_geoPandas = cities_geoPandas.drop(['lat', 'lng'], axis=1)
    #cities_geoPandas = cities_geoPandas.rename(columns={date:'newc_count'})
    return cities_geoPandas

def get_states_geoJson():
    states_geoJson = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
    ).json()
    return states_geoJson
    
def get_state_level_data(df, date_columns, metric_name):
    # get monthly value of each region by taking mean
    df['metric_mean_over_time_bystate'] = df[date_columns].mean(axis=1)
    df['metric_per_pop_mean_over_time_bystate'] = df['metric_mean_over_time_bystate'] / df['population'] *10000
    if metric_name == 'newc_count':
        state_level_data = df.groupby(['StateName']) \
                .apply(lambda x: pd.Series({
                    metric_name: x['metric_mean_over_time_bystate'].sum(),
                    'newc_count_per_pop': x['metric_mean_over_time_bystate'].sum() / x['population'].sum()*10000,
                    'Population': x['population'].sum(),
                    'Density': x['density'].mean(),
                    })).reset_index()
    elif metric_name == 'newc_price':
        state_level_data = df.groupby(['StateName']) \
                .apply(lambda x: pd.Series({
                    metric_name: x['metric_mean_over_time_bystate'].mean(),
                    'Population': x['population'].sum(),
                    'Density': x['density'].mean(),
                    })).reset_index()

    return state_level_data
    
def plot_map(states_geoJson, state_level_data, state_level_data_price, cities_geoPandas):
    m = folium.Map(location=[38, -102], zoom_start=4, scrollWheelZoom=False)

    # add color layer to the map
    threshold_scale = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 5, 10, 27]
    choropleth = folium.Choropleth(
        geo_data=states_geoJson,
        name="choropleth",
        data=state_level_data,
        columns=["StateName",  "newc_count_per_pop"],
        key_on="feature.id",
        fill_color="PuBuGn",
        threshold_scale=threshold_scale,
        #bins=8,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="new construction count/population",
        highlight=True,
    ).add_to(m)

    #choropleth.geojson.add_to(m)

    # add tooltip when hover the mouse over
    # add newc_count into geojson data in order to show it
    for feature in choropleth.geojson.data['features']:
        state_name = feature['id']
        if state_name in state_level_data['StateName'].tolist():
            newc_count = state_level_data.loc[state_level_data['StateName']==state_name, 'newc_count'].values[0]
            newc_count_per_pop = state_level_data.loc[state_level_data['StateName']==state_name, 'newc_count_per_pop'].values[0]
            feature['properties']['newc_count'] = f'State monthly newc count: {newc_count:.2f}'
            feature['properties']['newc_count_per_pop'] = f'State monthly newc count/population percentage: {newc_count_per_pop:.2f}bps'
        if state_name in state_level_data_price['StateName'].tolist():
            newc_price = state_level_data_price.loc[state_level_data_price['StateName']==state_name, 'newc_price'].values[0]
            feature['properties']['newc_price'] = f'State Avg newc median price: {newc_price:.2f}'
        

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name', 'newc_count_per_pop', 'newc_count',  'newc_price'], labels=False)
    )
    
    # add cities
    folium.GeoJson(
        cities_geoPandas,
        name="Subway Stations",
        marker=folium.Circle(radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1),
        tooltip=folium.GeoJsonTooltip(fields=["RegionName", 'newc_count_per_pop', 'newc_count', 'newc_price', 'population','density']), 
        popup=folium.GeoJsonPopup(fields=["RegionName", 'newc_count_per_pop','newc_count', 'newc_price', 'population','density']),  
        style_function=lambda x: {
            "radius": (x['properties']['newc_count_per_pop'])*4000,
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
    st.caption(PAGE_SUB_TITLE0)
    st.caption(PAGE_SUB_TITLE)
    st.markdown("All the information provided below are within the time range selected here:")

    #Load Data
    df_info = pd.read_csv('data/uscities.csv')
    # count data
    df_newc_count = pd.read_csv('data/new_construction/Metro_new_con_sales_count_raw_uc_sfrcondo_month.csv')
    date_columns = pd.to_datetime(df_newc_count.columns, errors='coerce', format="%Y-%m-%d")
    is_date_column = np.array([isinstance(col, pd.Timestamp) for col in date_columns ])
    df_newc_count_states = df_newc_count[df_newc_count['RegionType']=='msa'].merge(df_info[['RegionName', 'population','density','lng', 'lat']], on='RegionName', how='left')
    df_newc_count_US = df_newc_count.loc[df_newc_count['RegionType']=='country', df_newc_count.columns[is_date_column]].T.reset_index(drop=False).rename(columns={0:'newc_count', 'index':'Date'})
    #st.write(df_newc_count_states[['population','density','lng', 'lat']].isna().sum())
    # median price data
    df_newc_price = pd.read_csv('data/new_construction/Metro_new_con_median_sale_price_uc_sfrcondo_month.csv')
    df_newc_price_states = df_newc_price[df_newc_price['RegionType']=='msa'].merge(df_info[['RegionName', 'population','density','lng', 'lat']], on='RegionName', how='left')
    df_newc_price_US = df_newc_price.loc[df_newc_price['RegionType']=='country', df_newc_price.columns[is_date_column]].T.reset_index(drop=False).rename(columns={0:'newc_price', 'index':'Date'})


    start_date, end_date = display_date_filter(df_newc_count_US)
    #state = display_state_filter(df_newc_count_states, 'StateName')
    # count
    subset_df_newc_count_states, date_columns_ls = df_time_filter(df_newc_count_states, start_date, end_date, 'wide')
    subset_df_newc_count_US = df_time_filter(df_newc_count_US, start_date, end_date, 'long')

    # price
    subset_df_newc_price_states, date_columns_ls = df_time_filter(df_newc_price_states, start_date, end_date, 'wide')
    subset_df_newc_price_US = df_time_filter(df_newc_price_US, start_date, end_date, 'long')

    cities_geoPandas = get_cities_data(subset_df_newc_count_states, subset_df_newc_price_states, date_columns_ls)
    states_geoJson = get_states_geoJson()
    state_level_data = get_state_level_data(subset_df_newc_count_states, date_columns_ls, 'newc_count')
    state_level_data_price = get_state_level_data(subset_df_newc_price_states, date_columns_ls, 'newc_price')


    # Create two columns with width ratios 3:1
    col1, col2, col3 = st.columns([24, 1, 7])
    # Add content to the first column (3/4 page width)
    with col1:
        st.subheader("Averaged Monthly New Construction Count/Population Ratio(bps) Across Regions")
        st.caption("Deeper green represents higher ratio for States, \nlarger circle size represents higher ratio for MSAs")
        # Add your content here for the main column
        state, state_name, city = plot_map(states_geoJson, state_level_data, state_level_data_price, cities_geoPandas)
        if city:
            state = subset_df_newc_count_states.loc[subset_df_newc_count_states['StateName']==state]['StateName'].values[0]

    with col3:
        st.subheader(" ")
        # USA
        if state == 'USA':
            US_newc_count = subset_df_newc_count_US['newc_count'].mean()
            US_newc_price = subset_df_newc_price_US['newc_price'].mean()
            st.metric(label="COUNTRY", value="USA")
            st.metric(label='Monthly New Construc Count/Population', value=f"{US_newc_count/341814420*10000:,.2f}bps")
            st.metric(label="Monthly New Construc Count", value=f"{US_newc_count:,.2f}")
            st.metric(label="POPULATION", value="341,814,420")
            st.metric(label="MEDIAN PRICE", value=f"{US_newc_price:,.2f}")
            st.metric(label="DENSITY / Km2", value=f"{37.1:,.2f}")
        else:
            if city == '':
                # State
                #df = subset_df_newc_count_states[(subset_df_newc_count_states['StateName']==state)]
                state_data = state_level_data[state_level_data['StateName']==state]
                
                st.metric(label="STATE", value=state_name)
                st.metric(label='Monthly New Construc Count/Population', value=f"{state_data['newc_count_per_pop'].values[0]:,.2f}bps")
                st.metric(label="Monthly New Construc Count", value=f"{state_data['newc_count'].values[0]:,.2f}")
                st.metric(label="POPULATION", value=f"{state_data['Population'].values[0]:,.2f}")
                if state in state_level_data_price['StateName'].tolist():
                    state_data_price = state_level_data_price[state_level_data_price['StateName']==state]
                    st.metric(label="MEDIAN PRICE", value=f"{state_data_price['newc_price'].values[0]:,.2f}")
                st.metric(label="DENSITY / Km2", value=f"{state_data['Density'].values[0]:,.2f}")
            else:
                # city
                city_data = subset_df_newc_count_states.loc[(subset_df_newc_count_states['RegionName']==city) &\
                                                 (subset_df_newc_count_states['StateName']==state)]
                # calculate monthly newc count for region
                city_monthly_newc_count = city_data[date_columns_ls].mean(axis=1).values[0]
                st.metric(label="CITY", value=city)
                st.metric(label='Monthly New Construc Count/Population', value=f"{city_data['metric_per_pop_mean_over_time_bystate'].values[0]:,.2f}bps")
                st.metric(label="Monthly New Construc Count", value=city_data['metric_mean_over_time_bystate'])
                st.metric(label="POPULATION", value=f"{city_data['population'].values[0]:,.2f}")
                if state in state_level_data_price['StateName'].tolist():
                    city_data_price = subset_df_newc_price_states.loc[(subset_df_newc_price_states['RegionName']==city) &\
                                                 (subset_df_newc_price_states['StateName']==state)]
                    st.metric(label="MEDIAN PRICE", value=f"{city_data_price[date_columns_ls].mean(axis=1).values[0]:,.2f}")
                st.metric(label="DENSITY / Km2", value=f"{city_data['density'].values[0]:,.2f}")

    #Display Filters and Map

    # top 10 regions bar chart
    col1, col2, col3 = st.columns([15, 2, 15])
    if state == 'USA':
        # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
        df_sorted_head = state_level_data.sort_values(by='newc_count', ascending=False).head(10)
        df_sorted_head_ratio = state_level_data.sort_values(by='newc_count_per_pop', ascending=False).head(10)

        with col1:
            # Create an Altair bar chart
            chart = bar_chart(data=df_sorted_head_ratio, xmetric_name='newc_count_per_pop', ymetric_name='StateName', color_name='#31a354', order='descending',
                              xaxis_label='New Construction Count/Population', yaxis_label='State', chart_title='Top 10 States by new construction count/population(bps)')
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

        with col3:
            # Create an Altair bar chart
            chart = bar_chart(data=df_sorted_head, xmetric_name='newc_count', ymetric_name='StateName', color_name='#99d8c9', order='descending',
                              xaxis_label='New Construction Count', yaxis_label='State', chart_title='Top 10 States by new construction count')
            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

    else:
        if city == '':
            # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
            cities_data_one_state = subset_df_newc_count_states.loc[subset_df_newc_count_states['StateName']==state]
            df_sorted_head = cities_data_one_state.sort_values(by='metric_mean_over_time_bystate', ascending=False).head(10)
            df_sorted_head_ratio = cities_data_one_state.sort_values(by='metric_per_pop_mean_over_time_bystate', ascending=False).head(10)

            with col1:
                # Create an Altair bar chart
                chart = bar_chart(data=df_sorted_head_ratio, xmetric_name='metric_per_pop_mean_over_time_bystate', ymetric_name='RegionName', 
                                  color_name='#31a354', order='descending',
                                  xaxis_label='New Construction Count/Population', yaxis_label='MSA', chart_title='Top 10 MSAs by new construction count/population(bps)')
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)

            with col3:
                # Create an Altair bar chart
                chart = bar_chart(data=df_sorted_head, xmetric_name='metric_mean_over_time_bystate', ymetric_name='RegionName', 
                                  color_name='#99d8c9', order='descending',
                                  xaxis_label='New Construction Count', yaxis_label='MSA', chart_title='Top 10 MSAs by new construction count')
                # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True)



    # time series line chart
    caption_line = " average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD"
    if state == 'USA':
        st.subheader(state + ' Monthly New Construction Count Over Time')
        st.caption(state + caption_line)
        chart = line_chart(df=subset_df_newc_count_US, subset_df_newc_count_US=subset_df_newc_count_US, metric_name='newc_count', yaxis_label='New Construction Count')
        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)
    
    else:
        if city == '':
            st.subheader(state_name + ' Monthly New Construction Count Over Time')
            st.caption(state_name + caption_line)
            df = subset_df_newc_count_states[(subset_df_newc_count_states['StateName']==state)][date_columns_ls].mean(axis=0)
            df = df.reset_index(drop=False).rename(columns={'index':'Date', 0: 'newc_count'})
            
        else:
            st.subheader(city + ' Monthly New Construction Count Over Time')
            st.caption(city + caption_line)
            df = subset_df_newc_count_states.loc[(subset_df_newc_count_states['RegionName']==city) &\
                                            (subset_df_newc_count_states['StateName']==state)][date_columns_ls]
            df = df.T.reset_index(drop=False)
            df.columns = ['Date', 'newc_count']       
        #st.line_chart(data = df, x='Date', y='newc_count', height=250, use_container_width=True)
        chart = line_chart(df=df, subset_df_newc_count_US=subset_df_newc_count_US, metric_name='newc_count', yaxis_label='New Construction Count')
        st.altair_chart(chart, use_container_width=True)



if __name__ == "__main__":
    

    main()
