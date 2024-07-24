import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px

energy_columns = ['renewables_electricity', 'coal_electricity', 'gas_electricity', 'oil_electricity', 'nuclear_electricity', 'wind_electricity', 'solar_electricity', "hydro_electricity"]
# Load and prepare data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url)
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df.rename(columns={
        "renewables_electricity": "Renewables",
        "hydro_electricity": "Hydro",
        "solar_electricity": "Solar",
        "wind_electricity": "Wind",
        "coal_electricity": 'Coal',
        'gas_electricity': "Gas",
        'oil_electricity': "Oil",
        'nuclear_electricity' : "Nuclear"
    }, inplace=True)

    df = df[~df.country.isin([
        "G20 (Ember)",
        "OECD (Ember)",
        "Asia (Ember)",
        "G7 (Ember)",
        "OECD (EI)",
        "Asia Pacific (EI)",
        "High-income countries",
        "Europe (Ember)",
        "Non-OECD (EI)",
        "Asia",
        "North America (Ember)",
        "Upper-middle-income countries",
        "Europe (EI)",
        "Oceania (Ember)",
        "ASEAN (Ember)",
        "Lower-middle-income countries", 
        "European Union (27)", 
        "North America (EI)"
    ])]
    
    return df.sort_values('year')


@st.cache_data
def load_world_data(map_data):
    world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.dbf")
# data processing
    world.rename(columns={"ADMIN": "country"}, inplace=True)
# replace country name to fit the other dataset
    world.country = world.country.replace(
        ["United States of America", "Democratic Republic of the Congo", "Republic of the Congo", "United Republic of Tanzania", "The Bahamas", "Czechia", "eSwatini", "Republic of Serbia"], 
        ["United States", "Democratic Republic of Congo", "Congo", "Tanzania", "Bahamas", "Czechoslovakia", "Eswatini", "Serbia"]
    )

    # world.SOV_A3 = world.SOV_A3.replace(["US1", "CH1", "FR1", "KA1", "GB1", "NZ1", "AU1"], ["USA", "CHN", "FRA", "KAZ", "GBR", "NZL", "AUS"])
    # latest_year = recent_data['year'].max() - pd.Timedelta(days=365*2)
    # latest_data = recent_data[recent_data['year'] == latest_year]
    world = world.merge(map_data, on=['country'])
    return world


@st.cache_data
def get_map_plot(energy_type_map, year_map):
    match energy_type_map:
        case "Solar":
            color_scale = "Oranges"
        case "Wind":
            color_scale = "Greens"
        case "Hydro":
            color_scale = "Blues"
        case _ :
            color_scale = "Viridis"

    fig = px.choropleth(world, locations='ADM0_A3', color=energy_type_map, 
                    hover_name='country', projection='natural earth2', color_continuous_scale=color_scale,
                    title=f'{energy_type_map.replace("_", " ").title()} Production in {year_map}')

    return fig
df = load_data()


# Streamlit app
st.title('Global Energy Production Analysis')
# st.write(df.columns)
# st.write([col.split("_")[0].capitalize() for col in energy_columns])
# Sidebar for user input
st.sidebar.header('Filter Data')
start_year = st.sidebar.slider('Start Year', 1900, 2023, 1980)
end_year = st.sidebar.slider('End Year', 1900, 2022, 2023)

# Filter data based on selected years
filtered_df = df[(df['year'].dt.year >= start_year) & (df['year'].dt.year <= end_year)]

# Global trend plot
st.header('Global Energy Production Trends')
energy_types = st.multiselect(
    'Select Energy Types',
    [col.split("_")[0].capitalize() for col in energy_columns], 
    default=["Solar", "Wind", "Hydro"]
)

global_trend = filtered_df.groupby('year')[energy_types].mean()

fig = px.line(global_trend.reset_index(), x='year', y=energy_types,
              title='Global Energy Production Trends')
fig.update_yaxes(title="Electricity Production (TWh)")
st.plotly_chart(fig)

# Map of energy production
st.header('Energy Production Map')
energy_type_map = st.selectbox('Select Energy Type for Map', energy_types)
year_map = st.slider('Select Year for Map', start_year, end_year, end_year-10)


map_data = filtered_df[filtered_df['year'].dt.year == year_map]
world = load_world_data(map_data)

fig = get_map_plot(energy_type_map, year_map)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig)

# Top countries comparison
st.header('Top Countries Comparison')
n_countries = st.slider('Number of top countries to compare', 1, 20, 15)
top_countries = filtered_df.groupby('country')[energy_types[0]].mean().nlargest(n_countries).index.tolist()

st.write(top_countries)

top_countries_data = filtered_df[filtered_df['country'].isin(top_countries)]

fig = px.line(top_countries_data, x='year', y=energy_types[0], color='country',
              title=f'Top {n_countries} Countries in {energy_types[0].replace("_", " ").title()} Production')
st.plotly_chart(fig)

# Energy mix comparison
st.header('Energy Mix Comparison')
selected_country = st.selectbox('Select a Country', df['country'].unique(), index=None)
if selected_country == None:
    selected_country = "Italy"

country_data = filtered_df[filtered_df['country'] == selected_country]
# st.write(country_data)


energy_mix = country_data[energy_types]
energy_mix['year'] = country_data.year

fig = px.area(energy_mix.dropna(), x='year', y=energy_types,
              title=f'Energy Mix for {selected_country}')
fig.update_yaxes(title="Electricity Production (TWh)")
st.plotly_chart(fig)
# st.write(energy_mix)


st.write("""
This Streamlit app provides an interactive analysis of global energy production trends. 
You can use the sidebar and various selectors to customize the visualizations and explore different aspects of energy production data.
""")