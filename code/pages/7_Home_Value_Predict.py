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
# import requests
# from io import StringIO
import json


import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import Dense
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')



PAGE_TITLE = 'CITY CURRENT - PREDICTED HOME VALUE'
# PAGE_SUB_TITLE = 'Property value is '


def get_cities_data(df):

    cities_ts_data = US_state_city_filter(df, 'city')
    cities_data = cities_ts_data.loc[:,['RegionID', 'RegionName', 'StateName', 'population', 'HomeValue','density', 'lng', 'lat']]

    # cities_data.dropna(inplace=True)

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

   
def plot_map(states_geoJson, state_level_data, cities_geoPandas):
    m = folium.Map(location=[38, -102], zoom_start=4, scrollWheelZoom=False)

    # add color layer to the map
    choropleth = folium.Choropleth(
        geo_data=states_geoJson,
        name="choropleth",
        data=state_level_data,
        columns=["StateName", "HomeValue"],
        key_on="feature.id",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="HomeValue",
        highlight=True,
    )

    choropleth.geojson.add_to(m)

    # add tooltip when hover the mouse over
    # add HomeValue into geojson data in order to show it
    for feature in choropleth.geojson.data['features']:
        state_name = feature['id']
        HomeValue = state_level_data.loc[state_level_data['StateName']==state_name, 'HomeValue'].values[0]
        feature['properties']['HomeValue'] = f'State Avg HomeValue: {HomeValue:.2f}'
        

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name', 'HomeValue'], labels=False)
    )
    
    
    # add cities
    folium.GeoJson(
        cities_geoPandas,
        name="Subway Stations",
        marker=folium.Circle(radius=4, fill_color="orange", fill_opacity=0.4, color="black", weight=1),
        tooltip=folium.GeoJsonTooltip(fields=["RegionName",  'HomeValue', 'population', 'density']), # 
        popup=folium.GeoJsonPopup(fields=["RegionName", 'HomeValue', 'population', 'density',]),  # 
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
    

# cities_data = pd.read_csv(data_loc + '/Metro_zhvi.csv')
saved_models_loc = 'data/models/'

def _forecast_(model, last12m_df):
    test_predictions = []
    if last12m_df['ZHVI Home value'].isna().any():
        return 'Need more data'

    current_batch = last12m_df.astype('float32').values.reshape((1, 12, 1))

    for i in range(70):

        # get the prediction value for the first batch
        current_pred = model.predict(current_batch, verbose=0).mean()

        # append the prediction into the array
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[[current_pred]]],axis=1)

    return test_predictions


def predict_price(last12m_df):
#     models = [ 'fianl_dense', 'fianl_cnn', 'final_lstm']
    models = ['fianl_cnn']
    dates = pd.date_range(start=last12m_df.index[-1], periods=71, freq='M')
    test_predictions = {'Date': dates[1:]}

    for model_name in models:
    #   model = load_model(saved_models_loc + f'{model_name}.h5')
      model = load_model('data/models/fianl_cnn.h5')  
      preds = _forecast_(model, last12m_df)
      test_predictions[model_name] = np.exp(np.array(preds))-1

    return test_predictions


# this function may need some edits,
# it should return a 12-row dataset with Date as index and 'ZHVI Home value' as the only column
# if same data as what I have used here was used no change needed.
def get_last12m_region_data(data, region):
    region_data = data.loc[data['RegionName'] == region]\
                             .T[-12:].reset_index(drop=False)
    region_data.columns = ['Date', 'ZHVI Home value']
    region_data['ZHVI Home value'] = np.log1p(region_data['ZHVI Home value'].astype('float32') )
    region_data.set_index('Date', inplace=True)
    return region_data

def prepare_chart_data(cities_data, REGION, predictions):
    # Prepare actual data
    actual_df = cities_data[cities_data['RegionName'] == REGION].copy()
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    # date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
    # Filter columns where the column name matches the date pattern
    date_columns = actual_df.columns[actual_df.columns.str.match(date_pattern, na=False)]
    region_data = actual_df[date_columns].copy().T.reset_index(drop=False)
    namedict = {}
    col_names = region_data.columns
    new_names = ['Date','HomeValue']
    namedict = {x:y for x,y in zip(col_names,new_names)}
    region_data.rename(columns = namedict,inplace=True )
    region_data['Type'] = 'Actual'

    # Prepare prediction data
    prediction_df = pd.DataFrame(predictions)
    prediction_df.rename(columns={'fianl_cnn':'HomeValue'},inplace=True)
    prediction_df['Type'] = 'predicted'
    
   
    # Concatenate actual and predicted data
    full_df = pd.concat([region_data, prediction_df])
    return full_df






def US_state_city_filter(df_HomeValue, location_level):
    # df_filtered = df_HomeValue.query("RegionType == location_level")

    df_filtered = df_HomeValue[df_HomeValue["RegionType"] == location_level].copy()
    # date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
    # Filter columns where the column name matches the date pattern
    date_columns = df_filtered.columns[df_filtered.columns.str.match(date_pattern, na=False)]

    df_filtered['HomeValue'] = df_filtered.loc[:,date_columns[-1]].values


    final_columns = ['RegionID','MSAName', 'RegionName', 'RegionType','city', 'state_id', 'StateName','state_name', 'lat','lng', 'population', 'density', 'HomeValue']

    return df_filtered.loc[:,final_columns]

def plot_home_value_chart(data):
    chart = alt.Chart(data).mark_line(point=True).encode(
        x='Date:T',  # T for temporal
        y=alt.Y('HomeValue:Q', title='Home Value'),  # Q for quantitative
        color='Type:N',  # N for nominal
        tooltip=['Date:T', 'HomeValue:Q', 'Type:N']
    ).properties(
        width='container',
        height=400
    ).configure_view(
        strokeWidth=0  # Remove border around chart
    ).configure_axis(
        gridColor='lightgray'  # Customize grid color
    ).interactive()





    return chart



def main():
    st.set_page_config(
     page_title=PAGE_TITLE,
     layout="wide",
     initial_sidebar_state="expanded",
    )
    st.title(PAGE_TITLE)
    # st.caption(PAGE_SUB_TITLE)
    # st.markdown("All the information provided below are within the time range selected here:")

    #Load Data
    df_HomeValue = pd.read_csv('data/home_value/US_State_City_Sum.csv')   
  

    cities_geoPandas = get_cities_data(df_HomeValue)
    # states_geoJson = get_states_geoJson()

    # Specify the file path
    file_path = 'data/StateGeoJason/output.geojson'

    # Write the GeoJSON data to a file
    with open(file_path, 'r') as file:
        states_geoJson = json.load(file)


    state_level_data = US_state_city_filter(df_HomeValue,'state')


    # Create two columns with width ratios 3:1
    col1, col2, col3 = st.columns([24, 1, 7])
    # Add content to the first column (3/4 page width)
    with col1:
        st.subheader("Current Home Value Across Regions")
        # st.caption("Deeper green represents higher ratio for States, \nlarger circle size represents higher ratio for MSAs")
        # Add your content here for the main column
        state, state_name, city = plot_map(states_geoJson, state_level_data, cities_geoPandas)
        if city:
            state = df_HomeValue.loc[df_HomeValue['StateName']==state]['StateName'].values[0]

    with col3:
        st.subheader(" ")
        # USA
        if state == 'USA':
            df_filtered =  US_state_city_filter(df_HomeValue,'US')

            US_HomeValue = df_filtered['HomeValue'].values[0]
            st.metric(label="COUNTRY...", value="USA")
            st.metric(label="AVG Home Value", value=f"${US_HomeValue:,.0f}")
            st.metric(label="POPULATION", value="341,814,420")
            st.metric(label="DENSITY / Km2", value=f"{37.1:,.1f}")
        else:
            if city == '':
                # State
                #df = subset_df_HomeValue_states[(subset_df_HomeValue_states['StateName']==state)]
                df_filtered =  US_state_city_filter(df_HomeValue,'state')

                state_data = df_filtered[df_filtered['StateName']==state]
                
                st.metric(label="STATE", value=state_name)
                st.metric(label="AVG Home Value", value=f"${state_data['HomeValue'].values[0]:,.0f}")
                st.metric(label="POPULATION", value=f"{state_data['population'].values[0]:,.0f}")
                st.metric(label="DENSITY / Km2", value=f"{state_data['density'].values[0]:,.1f}")
            else:
                # city
                df_filtered =  US_state_city_filter(df_HomeValue,'city')

                city_data = df_filtered.loc[(df_filtered['RegionName']==city) &\
                                                 (df_filtered['StateName']==state)]

                st.metric(label="CITY", value=city)
                st.metric(label="Home Value", value=f"${city_data['HomeValue'].values[0]:,.0f}")
                st.metric(label="POPULATION", value=f"{city_data['population'].values[0]:,.0f}")
                st.metric(label="DENSITY / Km2", value=f"{city_data['density'].values[0]:,.1f}")
             
    

    # time series line chart
    if state == 'USA':
        pass
   
    else:
        if city == '':
            pass
          
        else:
            # st.subheader(city + ' Monthly Home Value Over Time')
            # st.caption("US average value in 2018-4 - 2020-4 period as reference (show as purple dash line). \n Green dash line: average + 2*SD, Red dash line: average - 2*SD")
            cities_data = pd.read_csv('data/zhvi/Metro_zhvi.csv')
            cities_data.head()
            # REGION = 'Los Angeles, CA'

            REGION = city
            last12m = get_last12m_region_data(cities_data, REGION)
            predictions = predict_price(last12m)

            data_combined = prepare_chart_data(cities_data, REGION, predictions)


            # # Plot and display the chart
            home_value_chart = plot_home_value_chart(data_combined)
            # home_value_chart.display()
            st.altair_chart(home_value_chart, use_container_width=True)


if __name__ == "__main__":
    

    main()
