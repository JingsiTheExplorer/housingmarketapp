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

PAGE_TITLE = 'SALES PRICE vs. LIST PRICE'
PAGE_SUB_TITLE = "The ratio of the sales price to the listing price can indicate whether you are in a buyer's or seller's market. Changes in this ratio over time can signal shifts in the market, helping investors anticipate trends and adjust strategies accordingly. "

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
    

def get_cities_data(df, date_columns, metric_name):
    '''
      calculate the average of the given time range
    '''
    cities_ts_data = df[date_columns]

    cities_data = df.copy()
    cities_data[f'mean_{metric_name}'] = cities_ts_data.mean(axis=1)

    cities_data.dropna(inplace=True)

    # Create Point geometries from latitude and longitude
    geometry = [Point(xy) for xy in zip(cities_data['lng'], cities_data['lat'])]
    # Convert DataFrame to GeoDataFrame
    cities_geoPandas = gpd.GeoDataFrame(cities_data, geometry=geometry)
    # Set the CRS for the GeoDataFrame
    cities_geoPandas.crs = 'EPSG:4326'  # Assuming WGS84 coordinate reference system
    # Drop the latitude and longitude columns if needed
    cities_geoPandas = cities_geoPandas.drop(['lat', 'lng'], axis=1)
    #cities_geoPandas = cities_geoPandas.rename(columns={date:'DaysOnMarket'})
    return cities_geoPandas

def get_states_geoJson():
    states_geoJson = requests.get(
    "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/us_states.json"
    ).json()
    return states_geoJson
    
def get_state_level_data(df, date_columns):
    state_level_data = df.groupby(['StateName'])[date_columns].mean().reset_index(drop=False)
    state_level_data['metric_mean_over_time_bystate'] = state_level_data[date_columns].mean(axis=1)
    #st.write(state_level_data)

    return state_level_data
    
def plot_map(states_geoJson, state_level_data, cities_geoPandas):
    m = folium.Map(location=[38, -102], zoom_start=4, scrollWheelZoom=False)

    # add color layer to the map
    choropleth = folium.Choropleth(
        geo_data=states_geoJson,
        name="choropleth",
        data=state_level_data,
        columns=["StateName", 'metric_mean_over_time_bystate'],
        key_on="feature.id",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Sale / List Price Rate",
        highlight=True,
    ).add_to(m)
    
    # add tooltip when hover the mouse over
    # add DaysOnMarket into geojson data in order to show it
    for feature in choropleth.geojson.data['features']:
        state_name = feature['id']
        if state_name in state_level_data['StateName'].tolist():
            sale2list = state_level_data.loc[state_level_data['StateName']==state_name, 'metric_mean_over_time_bystate'].values[0]
            feature['properties']['metric_mean_over_time_bystate'] = f'State median sale/list price rate: {sale2list:.2f}%'
        

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name', 'metric_mean_over_time_bystate'], labels=False)
    )
    
    
    # add cities
    folium.GeoJson(
        cities_geoPandas,
        name="Subway Stations",
        marker=folium.Circle(radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1),
        tooltip=folium.GeoJsonTooltip(fields=["RegionName",  'mean_median_sale_to_list']), # 
        popup=folium.GeoJsonPopup(fields=["RegionName", 'mean_median_sale_to_list']),  # 
        style_function=lambda x: {
            "radius": (x['properties']['mean_median_sale_to_list'])*40000,
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
    

def display_date_filter(date_list):
    # filter the time range
    col1, col2, col3 = st.columns([10, 10, 4])
    date_list_reverse = date_list[::-1]
    start_date = col1.selectbox("Start Date", date_list_reverse, len(date_list_reverse)-1)
    end_date = col2.selectbox("End Date", date_list_reverse, 0)
    return start_date, end_date



def line_chart(df, region_name, date_columns_ls, ylabel, legend_name):
    # time series line chart

    #st.line_chart(data = subset_df_DaysOnMarket_US, x='Date', y='DaysOnMarket', height=250, use_container_width=True)
    # Create an Altair chart object
    # Create an Altair chart object

    df = df[[region_name] + date_columns_ls.tolist()]
    df = df.set_index(region_name).T.reset_index(drop=False).rename(columns={'index':'Date'})

    
    df_long = df.melt(id_vars='Date', var_name='Region', value_name='Value')
    #st.write(df_long.head())

    chart = alt.Chart(df_long).mark_line(point=True).encode(
    x=alt.X('yearmonth(Date):T', title=''),  # T indicates temporal data (date or time)
    y=alt.Y('Value:Q', title=ylabel),  # Q indicates quantitative data
    color=alt.Color('Region:N', legend=alt.Legend(title=legend_name)),  # N indicates nominal (categorical) data
    tooltip=['Date:T', 'Region:N', 'Value:Q']
    ).properties(
    title=' ',
    width=600,
    height=300
    )
    return chart

def bar_chart(data, region_level, metric_name, xmetric_name, ymetric_name, color_name, order, xaxis_label, yaxis_label, chart_title):
    if region_level == 'state':
        data.drop(columns='geometry', inplace=True)

    if metric_name == 'median_sale_to_list':
        chart = alt.Chart(data).mark_bar(color=color_name).encode(
        x=alt.X(f'{xmetric_name}:N', sort=alt.EncodingSortField(field=ymetric_name, order=order), title=xaxis_label),  # 'N' indicates nominal (categorical) data
        y=alt.Y(f'{ymetric_name}:Q', title=yaxis_label),  # 'Q' indicates quantitative data
        #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
        tooltip=[xmetric_name, ymetric_name]  # Show tooltip on hover
        ).properties(
        title=chart_title, 
        width='container',  # Make the chart responsive
        height=300  # Set a fixed height for the chart
        )
    else:
        chart = alt.Chart(data).mark_bar(color=color_name).encode(
        x=alt.X(f'{xmetric_name}:N', sort=alt.EncodingSortField(field=ymetric_name, order=order), title=xaxis_label),  # 'N' indicates nominal (categorical) data
        y=alt.Y(f'{ymetric_name}:Q', title=yaxis_label),  # 'Q' indicates quantitative data
        #color=alt.Color('StateName:N', legend=None),  # Color bars by state, remove legend
        tooltip=[xmetric_name, ymetric_name]  # Show tooltip on hover
        ).properties(
        title=chart_title, 
        width='container',  # Make the chart responsive
        height=300  # Set a fixed height for the chart
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
    df_info = pd.read_csv('data/uscities.csv')
    df = {}
    df['median_sale_price'] = pd.read_csv('data/sales_price/Metro_median_sale_price_uc_sfrcondo_sm_sa_month.csv')
    df['median_sale_to_list'] = pd.read_csv('data/sales_price/Metro_median_sale_to_list_uc_sfrcondo_sm_month.csv')
    df['pct_sold_abovelist'] = pd.read_csv('data/sales_price/Metro_pct_sold_above_list_uc_sfrcondo_sm_month.csv')
    df['pct_sold_belowlist'] = pd.read_csv('data/sales_price/Metro_pct_sold_below_list_uc_sfrcondo_sm_month.csv')

    date_columns = pd.to_datetime(df['median_sale_price'].columns, errors='coerce', format="%Y-%m-%d")
    is_date_column = np.array([isinstance(col, pd.Timestamp) for col in date_columns ])
    date_columns_ls = df['median_sale_price'].columns[is_date_column].tolist()

    df_metric_cities = {}
    df_metric_US = {}
    subset_df_metric_cities = {}
    subset_df_metric_US = {}
    start_date, end_date = display_date_filter(date_columns_ls)
    states_geoJson = get_states_geoJson()
    cities_geoPandas = {}
    state_level_data = {}
    for metric in ['median_sale_price', 'median_sale_to_list', 'pct_sold_abovelist', 'pct_sold_belowlist']:
        df_metric_cities[metric] = df[metric][df[metric]['RegionType']=='msa'].merge(df_info[['RegionName', 'population','density','lng', 'lat']], on='RegionName', how='left')
        df_metric_US[metric] = df[metric].loc[df[metric]['RegionType']=='country', date_columns_ls].T.reset_index(drop=False).rename(columns={0:metric, 'index':'Date'})
    
        subset_df_metric_cities[metric], sub_date_columns_ls = df_time_filter(df_metric_cities[metric], start_date, end_date, 'wide')
        subset_df_metric_US[metric] = df_time_filter(df_metric_US[metric], start_date, end_date, 'long')
    
        cities_geoPandas[metric] = get_cities_data(subset_df_metric_cities[metric], sub_date_columns_ls, metric)
        state_level_data[metric] = get_state_level_data(subset_df_metric_cities[metric], sub_date_columns_ls)

    # Create two columns with width ratios 3:1
    col1, col2, col3 = st.columns([24, 1, 7])
    # Add content to the first column (3/4 page width)
    with col1:
        st.subheader("Averaged Sale Price to List Price Ratio Across Regions")
        st.caption("Deeper color represents higher ratio for States, \nlarger circle size represents higher ratio for MSAs")
        # Add your content here for the main column
        state, state_name, city = plot_map(states_geoJson, state_level_data['median_sale_to_list'], cities_geoPandas['median_sale_to_list'])
        if city:
            state = subset_df_metric_cities['median_sale_to_list'].loc[subset_df_metric_cities['median_sale_to_list']['StateName']==state]['StateName'].values[0]

    with col3:
        st.subheader(" ")
        # USA
        if state == 'USA':
            US_side_info = {}
            for metric in ['median_sale_price', 'median_sale_to_list', 'pct_sold_abovelist', 'pct_sold_belowlist']:
                if metric == 'median_sale_price':
                    US_side_info[metric] = subset_df_metric_US[metric][metric].mean()
                else:
                    US_side_info[metric] = subset_df_metric_US[metric][metric].mean()*100
            st.metric(label="COUNTRY", value="USA")
            st.metric(label="SALE/LIST PRICE", value=f"{US_side_info['median_sale_to_list']:,.2f}%")
            st.metric(label="SOLD ABOVE LIST PCT", value=f"{US_side_info['pct_sold_abovelist']:,.2f}%")
            st.metric(label="SOLD BELOW LIST PCT", value=f"{US_side_info['pct_sold_belowlist']:,.2f}%")
            st.metric(label="MEDIAN SALE PRICE", value=f"${US_side_info['median_sale_price']:,.2f}")
        else:
            if city == '':
                # State
                state_side_info = {}
                for metric in ['median_sale_price', 'median_sale_to_list', 'pct_sold_abovelist', 'pct_sold_belowlist']:
                    if metric == 'median_sale_price':
                        state_side_info[metric] = state_level_data[metric].loc[state_level_data[metric]['StateName']==state, 'metric_mean_over_time_bystate'].values[0]
                    else:
                        state_side_info[metric] = state_level_data[metric].loc[state_level_data[metric]['StateName']==state, 'metric_mean_over_time_bystate'].values[0]*100
                st.metric(label="STATE", value=state_name)
                st.metric(label="SALE/LIST PRICE", value=f"{state_side_info['median_sale_to_list']:,.2f}%")
                st.metric(label="SOLD ABOVE LIST PCT", value=f"{state_side_info['pct_sold_abovelist']:,.2f}%")
                st.metric(label="SOLD BELOW LIST PCT", value=f"{state_side_info['pct_sold_belowlist']:,.2f}%")
                st.metric(label="MEDIAN SALE PRICE", value=f"${state_side_info['median_sale_price']:,.2f}")

            else:
                # city
                city_side_info = {}
                for metric in ['median_sale_price', 'median_sale_to_list', 'pct_sold_abovelist', 'pct_sold_belowlist']:
                    if metric == 'median_sale_price':
                        city_side_info[metric] = cities_geoPandas[metric].loc[(cities_geoPandas[metric]['RegionName']==city) &\
                                                 (cities_geoPandas[metric]['StateName']==state), f'mean_{metric}'].values[0]
                    else:
                        city_side_info[metric] = cities_geoPandas[metric].loc[(cities_geoPandas[metric]['RegionName']==city) &\
                                                 (cities_geoPandas[metric]['StateName']==state), f'mean_{metric}'].values[0] *100
                st.metric(label="CITY", value=city)
                st.metric(label="SALE/LIST PRICE", value=f"{city_side_info['median_sale_to_list']:,.2f}%")
                st.metric(label="SOLD ABOVE LIST PCT", value=f"{city_side_info['pct_sold_abovelist']:,.2f}%")
                st.metric(label="SOLD BELOW LIST PCT", value=f"{city_side_info['pct_sold_belowlist']:,.2f}%")
                st.metric(label="MEDIAN SALE PRICE", value=f"${city_side_info['median_sale_price']:,.2f}")

    #Display Filters and Map
    descrip_dic = {'median_sale_price': ('Median Sales Price', '#99d8c9'), 
                'median_sale_to_list' : ('Median Sale/List Price Rate', '#31a354'), 
                'pct_sold_abovelist': ('Sold above List Price Percentage', '#2c7fb8'), 
                'pct_sold_belowlist': ('Sold below List Price Percentage', '#fc9272')}
    
    # top 10 regions bar chart
    if state == 'USA':
        df_sorted_head = {}

        for metric in ['median_sale_price', 'median_sale_to_list', 'pct_sold_abovelist', 'pct_sold_belowlist']:
            st.subheader('Top 10 States by :blue[{}] from {} to {}'.format(descrip_dic[metric][0], start_date, end_date), divider='rainbow')
            # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
            df_sorted_head[metric] = state_level_data[metric].sort_values(by='metric_mean_over_time_bystate', ascending=False).head(10).reset_index(drop=True)
            
            # Create an Altair bar chart
            chart_bar = bar_chart(data=df_sorted_head[metric], region_level='country', metric_name=metric, xmetric_name='StateName', ymetric_name='metric_mean_over_time_bystate', color_name=descrip_dic[metric][1], order='descending',
                            xaxis_label='State', yaxis_label=f'Averaged {descrip_dic[metric][0]}', chart_title='')
            # Display the chart in Streamlit
            st.altair_chart(chart_bar, use_container_width=True)

            # Create an top states line chart
            chart_line = line_chart(df_sorted_head[metric], 'StateName', sub_date_columns_ls, descrip_dic[metric][0], 'State')
            # Display the chart in Streamlit
            st.altair_chart(chart_line, use_container_width=True)

    else:
        if city == '':
            cities_data_one_state = {}
            df_sorted_head = {}

            for metric in ['median_sale_price', 'median_sale_to_list', 'pct_sold_abovelist', 'pct_sold_belowlist']:
                st.subheader('Top 10 MSAs by :blue[{}] from {} to {}'.format(descrip_dic[metric][0], start_date, end_date), divider='rainbow')
                # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
                # Sort the DataFrame based on the 'ratio' column in descending order and select the top 10
                cities_data_one_state[metric] = cities_geoPandas[metric].loc[cities_geoPandas[metric]['StateName']==state]
                df_sorted_head[metric] = cities_data_one_state[metric].sort_values(by=f'mean_{metric}', ascending=False).head(10)
                
                # Create an Altair bar chart
                chart_bar = bar_chart(data=df_sorted_head[metric], region_level='state', metric_name=metric, xmetric_name='RegionName', ymetric_name=f'mean_{metric}', 
                                      color_name=descrip_dic[metric][1], order='descending',
                                xaxis_label='MSA', yaxis_label=f'Averaged {descrip_dic[metric][0]}', chart_title='')
                # Display the chart in Streamlit
                st.altair_chart(chart_bar, use_container_width=True)

                # Create an top states line chart
                chart_line = line_chart(df_sorted_head[metric], 'RegionName', date_columns_ls,  descrip_dic[metric][0], 'MSA')
                # Display the chart in Streamlit
                st.altair_chart(chart_line, use_container_width=True)

 



if __name__ == "__main__":
    

    main()
