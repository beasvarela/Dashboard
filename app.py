
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.decomposition import PCA


import dash
from dash import dcc, html, Input, Output, State

np.random.seed(42)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path_original = os.path.join(BASE_DIR, "DM_AIAI_CustomerDB.csv")
df_original = pd.read_csv(path_original)

path_scaled = os.path.join(BASE_DIR, "df_customer_scaled.csv") 
df_customer_scaled = pd.read_csv(path_scaled) # final dataframe with final solution labels

path_treated = os.path.join(BASE_DIR, "df_customer_treated.csv")
df_customer_treated = pd.read_csv(path_treated) # pre-processed dataframe with no scaling or encoding



df_customer_treated = df_customer_treated[df_customer_treated.index.isin(df_customer_scaled.index)]
# creating a visualization dataframe by copying df_customer_scaled and adding previously dropped features
df_vis = df_customer_scaled.copy()
df_vis['Gender'] = df_customer_treated['Gender']
df_vis['Income_not_scaled'] = df_customer_treated['Income']
df_vis['City'] = df_customer_treated['City']
df_vis['Education'] = df_customer_treated['Education']
df_vis['in_program'] = df_customer_treated['in_program']
df_vis['Marital Status'] = df_customer_treated['Marital Status']
df_vis['TotalFlights_not_scaled'] = df_customer_treated['TotalFlights']
df_vis['ActiveMonths_not_scaled'] = df_customer_treated['ActiveMonths']
df_vis['TotalDistanceKM_not_scaled'] = df_customer_treated['TotalDistanceKM']
df_vis['Recency_not_scaled'] = df_customer_treated['Recency']

# since we dropped 'Province or State' right in the beginning, we have to get it from the original dataframe
df_vis['Province or State'] = df_original.loc[df_original.index.isin(df_vis.index), 'Province or State']



# we apply PCA to the features that were used for the clustering solution
pca = PCA(n_components=3)


pca_feats =['AdjustedCLV', 'AnnualizedCLV', 'RedemptionRatio',
 'LoyaltyStatus_Ordinal', 'AverageFlightsPerActiveMonth',
 'MembershipDurationYears', 'TotalDistanceKM', 'AvgDistancePerFlight',
 'CompanionRatio', 'RatioActivePeriod','Recency']

pca_coords = pca.fit_transform(df_customer_scaled[pca_feats])

# adding each of the principal components to the visualization dataframe
df_vis['x'] = pca_coords[:, 0]
df_vis['y'] = pca_coords[:, 1]
df_vis['z'] = pca_coords[:, 2]


# getting umap coordinates
umap_ax = umap.UMAP(n_components=3, random_state=42, n_neighbors=15,min_dist=0.1, metric='euclidean')
umap_coords = umap_ax.fit_transform(df_customer_scaled[pca_feats])
df_vis['umap_x'] = umap_coords[:, 0]
df_vis['umap_y'] = umap_coords[:, 1]
df_vis['umap_z'] = umap_coords[:, 2]
df_vis['in_program'].unique()


app = dash.Dash(__name__)
server = app.server   


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <style>

            /* ------------------------ Styling base containers of the dashboard: ---------------------------- */

            body { font-family: Arial; background: #EBDAD5; margin: 0; padding: 20px; }

            .dashboard-container { max-width: 1400px; margin: 0 auto; }

            .header { text-align: center; background: #FFFFFF; color: #242424; padding: 10px; border-radius: 15px; margin-bottom: 25px; }

            .filters-section { background: #FFFFFF; padding: 10px; border-radius: 15px; margin-bottom: 25px; }

            .filters { display: flex; flex-direction: row; flex-wrap: wrap; gap: 10px 0; }

            .pick-filters { margin-bottom: 20px; margin-left: 10px; padding: 5px; font-size: 5; }



        
            /* ------------------------------ Styling the rounded buttons for filtering: -------------------- */

            .rounded-button {width: 150px; background: #CFB5FF; color: #FFFFFF; padding: 12px 15px; border-radius: 15px; border: 1px solid #CFB5FF; cursor: pointer; font-size: 15px; font-weight: 600; margin-right: 15px; margin-left: 20px; margin-bottom: 5px; }

            .rounded-button:hover { background: #FFFFFF; color: #CFB5FF; border-color: #CFB5FF; }

            .filter-group {position: relative; display: inline-block;}

            .dropdown { display: none; position: absolute; top: 100%; margin-left: 20px; background: #FFFFFF; border-radius: 15px; border: 1px solid #CFB5FF; 
            padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); z-index: 100;}

            .filter-group:hover .dropdown {display: block;}

            .dropdown_option {width: 70%; transition: all 0.3s ease; background: #FFFFFF; color: #242424; padding: 8px 8px; border-radius: 15px; cursor: pointer; font-size: 15px; font-weight: 600; margin-top: 5px; margin-left: 5px; margin-bottom: 5px; }

            #income-dropdown, #flights-dropdown, #distance-dropdown, #months-dropdown, #recency-dropdown {width: 300px; margin-left: 5px;}

            #income-dropdown .rc-slider, #flights-dropdown .rc-slider, #distance-dropdown .rc-slider, #months-dropdown .rc-slider, #recency-dropdown .rc-slider { width: 100% !important;}

            #city-dropdown, #state-dropdown {max-height: 300px; overflow-y: auto; padding-right: 5px; }

            .dropdown_option:hover { background: #CFB5FF; color: white; transform: translateX(5px); }

            .dropdown_option.selected { background: #CFB5FF; color: white; font-weight: bold; }

            .graph-container { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
        document.addEventListener('click', function(e) {
            if(e.target.classList.contains('dropdown_option')) { e.target.classList.toggle('selected') ;}
        });
        </script>
    </body>
</html>
'''




app.layout = html.Div([ 


    html.Div([html.H1("Cluster Visualization Dashboard")], className="header"),

    dcc.Store(id="filter-state",
    data={"gender": [],"marital": [], "education": [], "city": [], "state": [], "program": []}),


    html.Div([
        html.H3("Pick your filters:", className="pick-filters"),

        html.Div([

            # Gender Filter
            html.Div([ 
                html.Button("Gender", id="gender-button", className="rounded-button", n_clicks=0),
                html.Div([
                html.P("Male", className="dropdown_option", id="male-option", n_clicks=0),
                html.P("Female", className="dropdown_option", id="female-option", n_clicks=0),
                html.P("All", className="dropdown_option", id="all-gender-option", n_clicks=0),
            ], className="dropdown", id="gender-dropdown")
            ], className = "filter-group"),

            # Income Filter
            html.Div([ 
                html.Button("Income", id="income-button", className="rounded-button", n_clicks=0),
                html.Div([
                        dcc.RangeSlider(id='income-slider', min=df_customer_treated['Income'].min(), max=df_customer_treated['Income'].max(), step=1000,
                        value=[df_customer_treated['Income'].min(), df_customer_treated['Income'].max()],
                        marks={ int(df_customer_treated['Income'].min()): str(int(df_customer_treated['Income'].min())),
                        int(df_customer_treated['Income'].max()): str(int(df_customer_treated['Income'].max()))},
                        tooltip={"placement": "bottom", "always_visible": True}, allowCross=False)], className="dropdown", id="income-dropdown")], className = "filter-group"),
            
            # Marital Status Filter
            html.Div([
                html.Button("Marital Status", id="marital-button", className="rounded-button", n_clicks=0),
                html.Div(
                [html.P(f"{stat}", className="dropdown_option", id=f"{stat.lower().replace(' ', '-')}-option", n_clicks=0) 
            for stat in df_original['Marital Status'].unique().tolist()] + [ 
            html.P("All", className="dropdown_option", id="all-marital-option", n_clicks=0)], className="dropdown", id="marital-dropdown"
            )], className="filter-group"),

            # Education Filter
            html.Div([
                html.Button("Education", id="education-button", className="rounded-button", n_clicks=0),
                html.Div(
                [html.P(f"{educ}", className="dropdown_option", id=f"{educ.lower().replace(' ', '-')}-option", n_clicks=0) 
            for educ in df_original['Education'].unique().tolist()] + [ 
            html.P("All", className="dropdown_option", id="all-education-option", n_clicks=0)], className="dropdown", id="education-dropdown"
            )], className="filter-group"),

            # State/Province Filter
            html.Div([
                html.Button("State/Province", id="state-button", className="rounded-button", n_clicks=0),
                html.Div(
                [html.P(f"{state}", className="dropdown_option", id=f"{state.lower().replace(' ', '-')}-option", n_clicks=0) 
            for state in df_original['Province or State'].unique().tolist()] + [ 
            html.P("All", className="dropdown_option", id="all-state-option", n_clicks=0)], className="dropdown", id="state-dropdown"
            )], className="filter-group"),


            # City Filter
            html.Div([
                html.Button("City", id="city-button", className="rounded-button", n_clicks=0),
                html.Div(
                [html.P(f"{city}", className="dropdown_option", id=f"{city.lower().replace(' ', '-').replace('.', '').replace(chr(39), '')}-option", n_clicks=0) 
            for city in df_original['City'].unique().tolist()] + [ 
            html.P("All", className="dropdown_option", id="all-city-option", n_clicks=0)], className="dropdown", id="city-dropdown"
            )], className="filter-group"),

            # Total Flights Filter
            html.Div([ 
                html.Button("Total Flights", id="flights-button", className="rounded-button", n_clicks=0),
                html.Div([
                        dcc.RangeSlider(id='flights-slider', min=df_customer_treated['TotalFlights'].min(), max=df_customer_treated['TotalFlights'].max(), step=1,
                        value=[df_customer_treated['TotalFlights'].min(), df_customer_treated['TotalFlights'].max()],
                        marks={ int(df_customer_treated['TotalFlights'].min()): str(int(df_customer_treated['TotalFlights'].min())),
                        int(df_customer_treated['TotalFlights'].max()): str(int(df_customer_treated['TotalFlights'].max()))},
                        tooltip={"placement": "bottom", "always_visible": True}, allowCross=False)], className="dropdown", id="flights-dropdown")], className = "filter-group"),
            
            # Total Distance Filter
            html.Div([ 
                html.Button("Total Distance", id="distance-button", className="rounded-button", n_clicks=0),
                html.Div([
                        dcc.RangeSlider(id='distance-slider', min=df_customer_treated['TotalDistanceKM'].min(), max=df_customer_treated['TotalDistanceKM'].max(), step=1,
                        value=[df_customer_treated['TotalDistanceKM'].min(), df_customer_treated['TotalDistanceKM'].max()],
                        marks={ int(df_customer_treated['TotalDistanceKM'].min()): str(int(df_customer_treated['TotalDistanceKM'].min())),
                        int(df_customer_treated['TotalDistanceKM'].max()): str(int(df_customer_treated['TotalDistanceKM'].max()))},
                        tooltip={"placement": "bottom", "always_visible": True}, allowCross=False)], className="dropdown", id="distance-dropdown")], className = "filter-group"),
            

            # Active Months Filter
            html.Div([ 
                html.Button("Active Months", id="months-button", className="rounded-button", n_clicks=0),
                html.Div([
                        dcc.RangeSlider(id='months-slider', min=df_customer_treated['ActiveMonths'].min(), max=df_customer_treated['ActiveMonths'].max(), step=1,
                        value=[df_customer_treated['ActiveMonths'].min(), df_customer_treated['ActiveMonths'].max()],
                        marks={ int(df_customer_treated['ActiveMonths'].min()): str(int(df_customer_treated['ActiveMonths'].min())),
                        int(df_customer_treated['ActiveMonths'].max()): str(int(df_customer_treated['ActiveMonths'].max()))},
                        tooltip={"placement": "bottom", "always_visible": True}, allowCross=False)], className="dropdown", id="months-dropdown")], className = "filter-group"),
            
            # Recency Filter
            html.Div([ 
                html.Button("Recency", id="recency-button", className="rounded-button", n_clicks=0),
                html.Div([
                        dcc.RangeSlider(id='recency-slider', min=df_customer_treated['Recency'].min(), max=df_customer_treated['Recency'].max(), step=1,
                        value=[df_customer_treated['Recency'].min(), df_customer_treated['Recency'].max()],
                        marks={ int(df_customer_treated['Recency'].min()): str(int(df_customer_treated['Recency'].min())),
                        int(df_customer_treated['Recency'].max()): str(int(df_customer_treated['Recency'].max()))},
                        tooltip={"placement": "bottom", "always_visible": True}, allowCross=False)], className="dropdown", id="recency-dropdown")], className = "filter-group"),
            
            # Enrollment Filter 
            html.Div([ 
                html.Button("Program Status", id="program-button", className="rounded-button", n_clicks=0),
                html.Div([
                html.P("Enrolled", className="dropdown_option", id="enrolled-option", n_clicks=0),
                html.P("Not Enrolled", className="dropdown_option", id="not-enrolled-option", n_clicks=0),
                html.P("All", className="dropdown_option", id="all-enrollment-option", n_clicks=0),
            ], className="dropdown", id="enrollment-dropdown")
            ], className = "filter-group"),

        
            ], className = "filters"
            )], className= "filters-section" ),


            html.Div([dcc.Graph(id="cluster-pca", style={"height": "700px"})], className="graph-container"),
            html.Div([dcc.Graph(id="cluster-umap", style={"height": "700px"})], className="graph-container"),

            html.Button("Download Dashboard HTML", id="download-html"),
            dcc.Download(id="html-download")

            ], className="dashboard-container") #end of layout

# helper function: when an option is selected the state of the filter lists changes
def toggle_value(lst, value):
    if value in lst:
        lst.remove(value)
    else:
        lst.append(value)
    return lst

@app.callback(
    Output("html-download", "data"),
    Input("download-html", "n_clicks"),
    prevent_initial_call=True
)
def export_dashboard(n):
    return dict(
        content=app.index_string,
        filename="dashboard.html"
    )

# whenever one of these inputs changes the update_plot function is called and the 3D plot is updated
@app.callback(
    Output("filter-state", "data"),

    # input structure: button option id, click counter (when one of the existing options is clicked, the click counter
    # increases by 1 and the input changes, triggering the callout)

    # gender inputs
    [Input("male-option", "n_clicks"), Input("female-option", "n_clicks"), Input("all-gender-option", "n_clicks"),
    # income input
    Input("income-slider", "value"),
    # marital status input
    Input("single-option", "n_clicks"), Input("married-option", "n_clicks"), Input("divorced-option", "n_clicks"), Input("all-marital-option", "n_clicks"),
    # education
    Input("high-school-or-below-option", "n_clicks"), Input("college-option", "n_clicks"), Input("bachelor-option", "n_clicks"), 
    Input("master-option", "n_clicks"),Input("doctor-option", "n_clicks"), Input("all-education-option", "n_clicks"), 
    # city inputs
    Input("toronto-option", "n_clicks"), Input("edmonton-option", "n_clicks"), Input("vancouver-option", "n_clicks"),
    Input("hull-option", "n_clicks"), Input("whitehorse-option", "n_clicks"), Input("trenton-option", "n_clicks"), Input("montreal-option", "n_clicks"),
    Input("dawson-creek-option", "n_clicks"), Input("quebec-city-option", "n_clicks"), Input("moncton-option", "n_clicks"),
    Input("fredericton-option", "n_clicks"), Input("ottawa-option", "n_clicks"), Input("tremblant-option", "n_clicks"),
    Input("calgary-option", "n_clicks"), Input("whistler-option", "n_clicks"), Input("thunder-bay-option", "n_clicks"),
    Input("peace-river-option", "n_clicks"), Input("winnipeg-option", "n_clicks"), Input("sudbury-option", "n_clicks"), Input("west-vancouver-option", "n_clicks"),
    Input("halifax-option", "n_clicks"), Input("london-option", "n_clicks"), Input("victoria-option", "n_clicks"),
    Input("regina-option", "n_clicks"), Input("kelowna-option", "n_clicks"), Input("st-johns-option", "n_clicks"),
    Input("kingston-option", "n_clicks"), Input("banff-option", "n_clicks"), Input("charlottetown-option", "n_clicks"),
    Input("all-city-option", "n_clicks"),
    # province/state inputs
    Input("ontario-option", "n_clicks"),Input("alberta-option", "n_clicks"),Input("british-columbia-option", "n_clicks"),
    Input("quebec-option", "n_clicks"),Input("yukon-option", "n_clicks"),Input("new-brunswick-option", "n_clicks"),
    Input("manitoba-option", "n_clicks"),Input("nova-scotia-option", "n_clicks"),Input("saskatchewan-option", "n_clicks"),
    Input("newfoundland-option", "n_clicks"),Input("prince-edward-island-option", "n_clicks"),Input("all-state-option", "n_clicks"),
    # program status input
    Input("enrolled-option", "n_clicks"), Input("not-enrolled-option", "n_clicks"), Input("all-enrollment-option", "n_clicks"),],
    State("filter-state", "data"), prevent_initial_call=True)


def update_filter_state(*args): # arguments come from the callout inputs in order

    state = args[-1] # state is the last argument
    ctx = dash.callback_context
    trigger = ctx.triggered_id

    # -------------- gender options ----------------------
    if trigger == "male-option":
        state["gender"] = toggle_value(state["gender"], "male")
    elif trigger == "female-option":
        state["gender"] = toggle_value(state["gender"], "female")
    elif trigger == "all-gender-option":
        state["geder"] = []

    # -------------- marital status options ----------------------
    elif trigger == "single-option":
        state["marital"] = toggle_value(state["marital"], "Single")
    elif trigger == "married-option":
        state["marital"] = toggle_value(state["marital"], "Married")
    elif trigger == "divorced-option":
        state["marital"] = toggle_value(state["marital"], "Divorced")
    elif trigger == "all-marital-option":
        state["marital"] = []

    # -------------- education options ----------------------
    elif trigger == "high-school-or-below-option":
        state["education"] = toggle_value(state["education"], "High School or Below")
    elif trigger == "college-option":
        state["education"] = toggle_value(state["education"], "College")
    elif trigger == "bachelor-option":
        state["education"] = toggle_value(state["education"], "Bachelor")
    elif trigger == "master-option":
        state["education"] = toggle_value(state["education"], "Master")
    elif trigger == "doctor-option":
        state["education"] = toggle_value(state["education"], "Doctor")
    elif trigger == "all-education-option":
        state["education"] = []
    


    # -------------- program options ----------------------
    elif trigger == "enrolled-option":
        state["program"] = toggle_value(state["program"], True)
    elif trigger == "not-enrolle-option":
        state["program"] = toggle_value(state["program"], False)
    elif trigger == "all-enrollment-option":
        state["program"] = []

    # -------------- city options ----------------------
    elif trigger == "toronto-option":
        state["city"] = toggle_value(state["city"], "Toronto")

    elif trigger == "edmonton-option":
        state["city"] = toggle_value(state["city"], "Edmonton")

    elif trigger == "vancouver-option":
        state["city"] = toggle_value(state["city"], "Vancouver")

    elif trigger == "hull-option":
        state["city"] = toggle_value(state["city"], "Hull")

    elif trigger == "whitehorse-option":
        state["city"] = toggle_value(state["city"], "Whitehorse")

    elif trigger == "trenton-option":
        state["city"] = toggle_value(state["city"], "Trenton")

    elif trigger == "montreal-option":
        state["city"] = toggle_value(state["city"], "Montreal")

    elif trigger == "dawson-creek-option":
        state["city"] = toggle_value(state["city"], "Dawson Creek")

    elif trigger == "quebec-city-option":
        state["city"] = toggle_value(state["city"], "Quebec City")

    elif trigger == "moncton-option":
        state["city"] = toggle_value(state["city"], "Moncton")

    elif trigger == "fredericton-option":
        state["city"] = toggle_value(state["city"], "Fredericton")

    elif trigger == "ottawa-option":
        state["city"] = toggle_value(state["city"], "Ottawa")

    elif trigger == "tremblant-option":
        state["city"] = toggle_value(state["city"], "Tremblant")

    elif trigger == "calgary-option":
        state["city"] = toggle_value(state["city"], "Calgary")

    elif trigger == "whistler-option":
        state["city"] = toggle_value(state["city"], "Whistler")

    elif trigger == "thunder-bay-option":
        state["city"] = toggle_value(state["city"], "Thunder Bay")

    elif trigger == "peace-river-option":
        state["city"] = toggle_value(state["city"], "Peace River")

    elif trigger == "winnipeg-option":
        state["city"] = toggle_value(state["city"], "Winnipeg")

    elif trigger == "sudbury-option":
        state["city"] = toggle_value(state["city"], "Sudbury")

    elif trigger == "west-vancouver-option":
        state["city"] = toggle_value(state["city"], "West Vancouver")

    elif trigger == "halifax-option":
        state["city"] = toggle_value(state["city"], "Halifax")

    elif trigger == "london-option":
        state["city"] = toggle_value(state["city"], "London")

    elif trigger == "victoria-option":
        state["city"] = toggle_value(state["city"], "Victoria")

    elif trigger == "regina-option":
        state["city"] = toggle_value(state["city"], "Regina")

    elif trigger == "kelowna-option":
        state["city"] = toggle_value(state["city"], "Kelowna")

    elif trigger == "st-johns-option":
        state["city"] = toggle_value(state["city"], "St. John's")

    elif trigger == "kingston-option":
        state["city"] = toggle_value(state["city"], "Kingston")

    elif trigger == "banff-option":
        state["city"] = toggle_value(state["city"], "Banff")

    elif trigger == "charlottetown-option":
        state["city"] = toggle_value(state["city"], "Charlottetown")

    elif trigger == "all-city-option":
        state["city"] = []

    # -------------- state options ----------------------
    elif trigger == "ontario-option":
        state["state"] = toggle_value(state["state"], "Ontario")

    elif trigger == "alberta-option":
        state["state"] = toggle_value(state["state"], "Alberta")

    elif trigger == "british-columbia-option":
        state["state"] = toggle_value(state["state"], "British Columbia")

    elif trigger == "quebec-option":
        state["state"] = toggle_value(state["state"], "Quebec")

    elif trigger == "yukon-option":
        state["state"] = toggle_value(state["state"], "Yukon")

    elif trigger == "new-brunswick-option":
        state["state"] = toggle_value(state["state"], "New Brunswick")

    elif trigger == "manitoba-option":
        state["state"] = toggle_value(state["state"], "Manitoba")

    elif trigger == "nova-scotia-option":
        state["state"] = toggle_value(state["state"], "Nova Scotia")

    elif trigger == "saskatchewan-option":
        state["state"] = toggle_value(state["state"], "Saskatchewan")

    elif trigger == "newfoundland-option":
        state["state"] = toggle_value(state["state"], "Newfoundland")

    elif trigger == "prince-edward-island-option":
        state["state"] = toggle_value(state["state"], "Prince Edward Island")

    elif trigger == "all-state-option":
        state["state"] = []


    return state


@app.callback(
    Output("cluster-pca", "figure"),
    Output("cluster-umap", "figure"),
    [
        Input("filter-state", "data"),
        Input("income-slider", "value"),
        Input("flights-slider", "value"),
        Input("months-slider", "value"),
        Input("distance-slider", "value"),
        Input("recency-slider", "value"),
    ]
)

def update_plot(state, income, flights, months, distance, recency):

    df = df_vis.copy()

    if state["gender"]:
        df = df[df["Gender"].isin(state["gender"])]


    if state["marital"]:
        df = df[df["Marital Status"].isin(state["marital"])]



    if state["education"]:
        df = df[df["Education"].isin(state["education"])]



    if state["city"]:
        df = df[df["City"].isin(state["city"])]



    if state["state"]:
        df = df[df["Province or State"].isin(state["state"])]



    if state["program"]:
        df = df[df["in_program"].isin(state["program"])]


    df = df[df["Income_not_scaled"].between(*income)]

    df = df[df["TotalFlights_not_scaled"].between(*flights)]

    df = df[df["ActiveMonths_not_scaled"].between(*months)]

    df = df[df["TotalDistanceKM_not_scaled"].between(*distance)]

    df = df[df["Recency_not_scaled"].between(*recency)]

    df = df.rename(columns={'Income_not_scaled': 'Income', 'TotalFlights_not_scaled': 'Total Flights',
                            'ActiveMonths_not_scaled': 'Active Months', 'TotalDistanceKM_not_scaled':'Total Distance (Km)',
                             'Recency_not_scaled': 'Recency (days)' })

    df['merged_labels'] = df['merged_labels'].astype(str)
    df['Recency (days)'] = df['Recency (days)'].round(0)

    fig_pca = px.scatter_3d(
        df,
        x="x", y="y", z="z",
        color="merged_labels",
        title=f"3D PCA Cluster Visualization (Showing {len(df)} points)",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        labels={"x": "PC1","y": "PC2", "z": "PC3", "merged_labels": "Cluster"},
        hover_data= ['Loyalty#', 'Income', 'City', 'Total Distance (Km)', 'Total Flights', 'Recency (days)']
    )

    fig_pca.update_traces(marker=dict(size=3))


    fig_umap = px.scatter_3d(
        df,
        x="umap_x", 
        y="umap_y", 
        z="umap_z",
        color="merged_labels",
        title=f"3D UMAP Cluster Visualization (Showing {len(df)} points)",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        labels={"umap_x": "UMAP1", "umap_y": "UMAP2", "umap_z": "UMAP3", "merged_labels": "Cluster"},
        hover_data=['Loyalty#', 'Income', 'City', 'Total Distance (Km)', 'Total Flights', 'Recency (days)']
    )
    fig_umap.update_traces(marker=dict(size=3))


    return fig_pca, fig_umap






if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    app.run_server(host="0.0.0.0", port=port)