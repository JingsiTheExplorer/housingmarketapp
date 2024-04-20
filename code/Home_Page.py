import streamlit as st 

PAGE_TITLE = "REAL ESTATE INVESTMENT SUPPORT TOOL"
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(PAGE_TITLE)

st.markdown("This real estate investment support app provides a comprehensive analysis tool for exploring home value appreciation, rental index increase, \
            supply and demand situation across different regions. Utilizing a combination of Python libraries such \
            as Pandas for data manipulation, Folium for interactive maps, and Streamlit for creating an easy-to-use web application, \
            this tool offers valuable insights through visual data representation. The application allows users to interactively explore various datasets, analyze trends in home values, and make informed decisions based on geographic data visualizations.")

st.markdown("""
    <style>
    .header-color {
        color: #FF6347; 
        font-size: 35px; 
        font-weight: bold; 
    }
    </style>
    """, unsafe_allow_html=True)
house_emoji = '\U0001F3E0'
waving_hand = '\U0001F44B'

st.markdown(f'<p class="header-color">{house_emoji}Historical Housing Market Status</p>', unsafe_allow_html=True)
st.subheader("1. Home Value Appreciation")
st.markdown("""<span style='color: blue;'>Home Value (Appreciation)</span>: Displays trends and forecasts in home values over time.""", unsafe_allow_html=True)
st.caption("Home value appreciation refers to the increase in the value of a property over time. The primary benefit of appreciation is the potential for capital gains. It affects both the long-term profitability and the strategic approach to property investment.")

st.subheader("2. Rental Profitability")
st.markdown("""<span style='color: blue;'>Rental Index Increase</span>: Tracks changes in rental costs.""", unsafe_allow_html=True)
st.caption("Higher rental indices increase generally indicate rising rental prices, which can lead to increased revenue for property owners. This makes real estate investments more attractive, as the potential return on investment (ROI) appears more favorable.")

st.subheader("3. Needed Effort to Sell")
st.markdown("""<span style='color: blue;'>Supply vs Demand</span>: Analyzes housing supply against market demand""", unsafe_allow_html=True)
st.caption("The dynamics between supply and demand levels play crucial roles in shaping real estate investment decisions. New listings and new pendings reflect the supply and demand situation of the real estate market to a certain extent. Here, we use (new pendings count / (inventory + new listing count)) ratios to compare the situations in different areas. The higher the ratio, the less effort to sell.")
st.markdown("""<span style='color: blue;'>Sales vs List Price</span>: Compares actual sales prices to listing prices.""", unsafe_allow_html=True)
st.caption("""The relationship between sales prices and list prices in real estate, often expressed as the "sale-to-list price ratio," is a critical indicator. A high sale-to-list price ratio, where sales prices are close to or exceed the list prices, indicates a seller’s market. This scenario suggests strong demand and competitive buying conditions.
Conversely, a lower ratio, where sales prices are significantly below list prices, points to a buyer's market, indicating weaker demand or an oversupply of properties.""")
st.markdown("""<span style='color: blue;'>Days on Market</span>: Shows the average time properties stay on the market before being sold.""", unsafe_allow_html=True)
st.caption("Days on Market (DOM) is a key real estate metric indicating how long a property remains listed before being sold or removed. A short DOM typically signals a strong seller's market with high demand and quick sales, prompting sellers to price aggressively and buyers to act swiftly to secure properties. Conversely, a long DOM suggests a buyer's market, potentially allowing buyers to negotiate lower prices due to reduced demand or seller urgency. ")
st.markdown("""<span style='color: blue;'>New Construction</span>: Provides statistics on newly constructed homes.""", unsafe_allow_html=True)
st.caption("An increase in new constructions usually signals a growing market with potential for economic expansion, attracting both residential and commercial investors. However, it can also lead to increased competition, possibly driving down prices for existing properties if supply outpaces demand. For investors, monitoring new construction trends helps in assessing future property value trajectories, market saturation levels.")

st.markdown(f'<p class="header-color">{waving_hand}Future Prediction', unsafe_allow_html=True)
st.markdown("""<span style='color: blue;'>Home Value Predict</span>: Offers predictive insights into future home values.""", unsafe_allow_html=True)