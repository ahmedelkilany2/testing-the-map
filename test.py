import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Photon
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import hashlib
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import re # For improved salary parsing

# Set up Photon geocoder 
geolocator = Photon(user_agent="vic_job_analysis_v2") # Updated user agent

# Google Sheets URL 
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1iFZ71DNkAtlJL_HsHG6oT98zG4zhE6RrT2bbIBVitUA/gviz/tq?tqx=out:csv"

# Currency conversion rate - AUD to USD
AUD_TO_USD = 0.67  

@st.cache_data(ttl=600)  
def load_data():
    """Load job location data from Google Sheets."""
    try:
        df = pd.read_csv(GOOGLE_SHEET_URL)
        df.columns = df.columns.str.strip().str.lower() 
        if "location" not in df.columns:
            st.error("⚠️ 'location' column missing in the dataset!")
            return None
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to load data: {str(e)}")
        return None

# Create a hash for each location 
def get_location_hash(location): # This function is defined but not explicitly used in the provided snippet. Keeping it.
    return hashlib.md5(location.encode()).hexdigest()

@st.cache_data(ttl=86400)  
def geocode_location(location):
    """Convert location names to latitude & longitude using Photon."""
    if pd.isna(location):
        return None, None
    try:
        full_location = f"{location}, Victoria, Australia"  
        location_data = geolocator.geocode(full_location, timeout=10)
        if location_data:
            return location_data.latitude, location_data.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        time.sleep(1) 
        try:
            full_location = f"{location}, Victoria, Australia"
            location_data = geolocator.geocode(full_location, timeout=15)
            if location_data:
                return location_data.latitude, location_data.longitude
        except Exception: # Catch generic exception on retry
            pass
    except Exception: # Catch generic exception on first try
        pass
    return None, None

def create_job_category_chart(df):
    """Create a horizontal bar chart for job categories."""
    if "job type" not in df.columns:
        return None
    
    job_counts = df["job type"].value_counts().reset_index()
    job_counts.columns = ["Category", "Count"]
    job_counts = job_counts.sort_values("Count", ascending=False)
    
    chart = alt.Chart(job_counts).mark_bar().encode(
        y=alt.Y('Category:N', sort='-x', title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X('Count:Q', title='Number of Job Postings'),
        color=alt.Color('Category:N', legend=None, scale=alt.Scale(scheme='category20')),
        tooltip=[alt.Tooltip('Category:N', title='Job Category'), alt.Tooltip('Count:Q', title='Number of Postings')]
    ).properties(
        title='Total Job Postings by Category'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    num_categories = len(job_counts)
    chart_height = max(200, num_categories * 20 + 60) # Adjusted height logic
    chart = chart.properties(height=chart_height)
    return chart

def categorize_job_titles(df):
    """Categorize jobs based on job titles."""
    title_column = None
    possible_columns = ["title", "job title", "jobtitle", "name"] # Added "name"
    
    for col in possible_columns:
        if col in df.columns:
            title_column = col
            break
    
    if title_column is None:
        st.warning("Could not find a suitable job title column for categorization (tried: 'title', 'job title', 'jobtitle', 'name').")
        return df # Return df as is, so 'title_category' column won't be created
        
    categories = {
        "Developer": ["develop", "programmer", "coding", "software engineer", "web dev", "full stack", "backend", "frontend", "sde"],
        "Data": ["data sci", "data analy", "machine learning", "ml", "ai", "analytics", "business intel", "data engineer"],
        "Management": ["manager", "head of", "director", "lead", "chief", "cto", "cio", "ceo", "vp ", "principal"],
        "Design": ["design", "ux", "ui", "user experience", "graphic", "product designer"],
        "Marketing": ["market", "growth", "seo", "content", "social media", "digital market", "brand"],
        "Finance": ["finance", "account", "tax", "audit", "financial", "analyst finan"],
        "HR": ["hr", "human resource", "recruit", "talent", "people", "personnel"],
        "Support": ["support", "help desk", "customer service", "technical support", "servicedesk"],
        "Sales": ["sales", "business develop", "account exec", "account manag", "customer success"],
        "Operations": ["operations", "project manag", "product manag", "scrum master", "program manag", "ops"],
        "Consulting": ["consultant", "advisory"],
        "System Admin/DevOps": ["sysadmin", "system admin", "devops", "sre", "cloud engineer", "infrastructure"],
        "Quality Assurance": ["qa", "quality assurance", "tester", "sdet"],
        "Security": ["security", "cybersecurity", "infosec"]
    }
    
    def assign_category(title):
        if pd.isna(title):
            return "Other"
        
        title_lower = str(title).lower()
        for category, keywords in categories.items():
            if any(keyword.lower() in title_lower for keyword in keywords):
                return category
        return "Other"
    
    df["title_category"] = df[title_column].apply(assign_category)
    return df

def create_job_title_category_chart(df):
    """Create a horizontal bar chart for job categories derived from titles."""
    if "title_category" not in df.columns:
        return None # title_category might not exist if title column wasn't found
    
    job_counts = df["title_category"].value_counts().reset_index()
    job_counts.columns = ["Category", "Count"]
    job_counts = job_counts.sort_values("Count", ascending=False)
    
    chart = alt.Chart(job_counts).mark_bar().encode(
        y=alt.Y('Category:N', sort='-x', title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X('Count:Q', title='Number of Job Postings'),
        color=alt.Color('Category:N', legend=None, scale=alt.Scale(scheme='tableau20')),
        tooltip=[alt.Tooltip('Category:N', title='Category (from Title)'), alt.Tooltip('Count:Q', title='Number of Postings')]
    ).properties(
        title='Job Categories Based on Title Keywords'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    num_categories = len(job_counts)
    chart_height = max(200, num_categories * 20 + 60) # Adjusted height logic
    chart = chart.properties(height=chart_height)
    return chart

def create_job_source_chart(df):
    """Create a pie chart showing job source distribution (Seek vs Other)."""
    source_col_found = False
    if "is from seek" in df.columns:
        df["is_seek_derived"] = df["is from seek"].apply( # Use a new column name
            lambda x: "Seek" if not pd.isna(x) and str(x).lower() in ("true", "1", "yes", "y") else "Other"
        )
        source_col_found = True
    else:
        source_identified = False
        columns_to_check = ["source", "url", "link", "job source", "platform"] # Added more potential column names
        for col in columns_to_check:
            if col in df.columns:
                # Check if the column actually contains string data for 'seek' detection
                if df[col].dtype == 'object' and df[col].str.contains('seek', case=False, na=False).any():
                    df["is_seek_derived"] = df[col].apply(
                        lambda x: "Seek" if not pd.isna(x) and "seek" in str(x).lower() else "Other"
                    )
                    source_identified = True
                    source_col_found = True
                    break
        
        if not source_identified: # Fallback: check all string columns if no specific column worked
            for col in df.select_dtypes(include='object').columns:
                if df[col].str.contains('seek', case=False, na=False).any():
                    df["is_seek_derived"] = df[col].apply(
                        lambda x: "Seek" if not pd.isna(x) and "seek" in str(x).lower() else "Other"
                    )
                    source_col_found = True
                    break
    
    if not source_col_found or "is_seek_derived" not in df.columns:
        # Data unavailable or column not created
        fig_data = pd.DataFrame({"Source": ["Unknown"], "Count": [len(df)]})
        title_text = "Job Source Distribution (Data Not Available)"
        colors = ['#cccccc']
    else:
        source_counts = df["is_seek_derived"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig_data = source_counts
        title_text = "Job Source Distribution"
        colors = ['#00B2A9', '#E4002B'] # Seek, Other
        if len(fig_data['Source']) == 1 and fig_data['Source'][0] == "Other": # if only "Other" is found
            colors = ['#E4002B']
        elif len(fig_data['Source']) == 1 and fig_data['Source'][0] == "Seek": # if only "Seek" is found
             colors = ['#00B2A9']


    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=fig_data['Source'],
        values=fig_data['Count'],
        hole=0.6,  
        textinfo='percent+value', 
        hoverinfo='label+value+percent',
        marker=dict(colors=colors),
        textfont=dict(size=16)
    ))
    fig.update_layout(
        title={'text': title_text, 'x':0.5, 'xanchor': 'center'},
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_job_status_chart(df):
    """Create a pie chart showing active vs inactive jobs."""
    status_col_found = False
    status_col_name_to_use = None

    if "is active" in df.columns:
        status_col_name_to_use = "is active"
    else:
        possible_status_columns = ["status", "active", "job_status", "job status"]
        for col in possible_status_columns:
            if col in df.columns:
                status_col_name_to_use = col
                break
    
    if status_col_name_to_use:
        def determine_status(value):
            if pd.isna(value): return "Unknown"
            val_str = str(value).lower()
            if any(term in val_str for term in ["true", "1", "yes", "active", "open", "live"]): return "Active"
            if any(term in val_str for term in ["false", "0", "no", "inactive", "closed", "expired"]): return "Inactive"
            return "Unknown"
        df["job_status_derived"] = df[status_col_name_to_use].apply(determine_status)
        status_col_found = True
    # Fallback: if no explicit status column, check date columns to infer 'Active'
    elif not status_col_found:
        date_cols_for_status = ["date", "posted", "created", "updated", "date posted", "last seen"]
        for col in date_cols_for_status:
            if col in df.columns:
                # A simple heuristic: if a date exists, assume active for this chart's purpose
                # More sophisticated logic would compare date to current date
                df["job_status_derived"] = df[col].apply(lambda x: "Active" if pd.notna(x) else "Unknown")
                if "Active" in df["job_status_derived"].unique(): # Check if it actually produced 'Active'
                    status_col_found = True
                    break
    
    if not status_col_found or "job_status_derived" not in df.columns:
        fig_data = pd.DataFrame({"Status": ["Unknown"], "Count": [len(df)]})
        title_text = "Job Status Distribution (Data Not Available)"
        status_colors_map = {"Unknown": "#9E9E9E"}
    else:
        status_counts = df["job_status_derived"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_data = status_counts
        title_text = "Job Status Distribution"
        status_colors_map = {"Active": "#4CAF50", "Inactive": "#F44336", "Unknown": "#9E9E9E"}

    colors_ordered = [status_colors_map.get(s, "#cccccc") for s in fig_data["Status"]]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=fig_data['Status'],
        values=fig_data['Count'],
        hole=0.6,  
        textinfo='percent+value',  
        hoverinfo='label+value+percent',
        marker=dict(colors=colors_ordered),
        textfont=dict(size=16)
    ))
    fig.update_layout(
        title={'text': title_text, 'x':0.5, 'xanchor': 'center'},
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_employment_type_charts(df):
    """Create donut charts for employment types."""
    charts = []
    
    # Full-time vs Part-time vs Other
    if "job type" in df.columns:
        def categorize_time(job_type_val):
            jt = str(job_type_val).lower()
            if 'full time' in jt or 'full-time' in jt: return 'Full-Time'
            if 'part time' in jt or 'part-time' in jt: return 'Part-Time'
            if 'casual' in jt: return 'Casual' # Added Casual
            return 'Other/Not Specified'
        
        df['time_category'] = df['job type'].apply(categorize_time)
        time_counts = df['time_category'].value_counts().reset_index()
        time_counts.columns = ['Category', 'Count']
        
        if not time_counts.empty:
            time_colors = {'Full-Time': '#3498db', 'Part-Time': '#e74c3c', 'Casual': '#f1c40f', 'Other/Not Specified': '#95a5a6'}
            ordered_colors = [time_colors.get(cat, '#bdc3c7') for cat in time_counts['Category']]

            time_fig = go.Figure(data=[go.Pie(
                labels=time_counts['Category'], 
                values=time_counts['Count'], 
                hole=0.6, 
                textinfo='percent+value',
                hoverinfo='label+value+percent',
                marker_colors=ordered_colors,
                textfont_size=14
            )])
            time_fig.update_layout(
                title={'text': "Employment Basis (Full-Time/Part-Time/Casual)", 'x':0.5, 'xanchor': 'center'},
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=50, b=0), height=300)
            charts.append(time_fig)
    
    # Contract vs Permanent vs Other
    contract_col_to_use = None
    contract_cols = ["contract type", "contract", "employment type", "job type"] # Broader search
    for col in contract_cols:
        if col in df.columns:
            contract_col_to_use = col
            break
    
    if contract_col_to_use:
        def categorize_contract(val):
            v = str(val).lower()
            if 'contract' in v or 'temporary' in v or 'fixed term' in v: return 'Contract/Temporary'
            if 'permanent' in v or 'ongoing' in v: return 'Permanent'
            return 'Other/Not Specified'
                
        df['contract_category'] = df[contract_col_to_use].apply(categorize_contract)
        contract_counts = df['contract_category'].value_counts().reset_index()
        contract_counts.columns = ['Category', 'Count']

        if not contract_counts.empty:
            contract_colors = {'Contract/Temporary': '#f39c12', 'Permanent': '#27ae60', 'Other/Not Specified': '#95a5a6'}
            ordered_colors = [contract_colors.get(cat, '#bdc3c7') for cat in contract_counts['Category']]

            contract_fig = go.Figure(data=[go.Pie(
                labels=contract_counts['Category'], 
                values=contract_counts['Count'], 
                hole=0.6, 
                textinfo='percent+value',
                hoverinfo='label+value+percent',
                marker_colors=ordered_colors,
                textfont_size=14
            )])
            contract_fig.update_layout(
                title={'text': "Contract Basis (Permanent/Contract)", 'x':0.5, 'xanchor': 'center'},
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=50, b=0), height=300)
            charts.append(contract_fig)
    
    return charts

def extract_salary_values(salary_text):
    """
    Extracts numerical salary values from a string. Improved version.
    Handles ranges (e.g., "80k - 100k", "80000 to 100000") and single values.
    Converts 'k' to thousands. Ignores common timeframes like "per hour", "p.a.".
    Returns a list of found numeric salaries (e.g., [80000, 100000] or [90000]).
    """
    if pd.isna(salary_text):
        return []
    
    text = str(salary_text).lower()
    # Remove "per hour", "per day", "per annum", "p.a" etc. to avoid parsing these as numbers
    text = re.sub(r'p\.?a\.?|per annum|per year|per day|per hour', '', text)
    # Remove currency symbols, commas specifically
    text = text.replace('$', '').replace(',', '')
    
    # Find numbers (potentially with 'k' and decimals)
    # Regex looks for numbers, optionally followed by 'k', separated by 'to' or '-' or just standalone
    # This is a simplified regex focusing on numbers and 'k'
    # It might capture multiple numbers if present, e.g. in "80k-90k plus 10k bonus" -> [80000, 90000, 10000]
    # We are interested in the base salary range.
    
    # Split by common range separators
    parts = re.split(r'\s+to\s+|\s*-\s*', text)
    
    numbers = []
    for part in parts:
        # Clean up each part, keeping only digits, '.', and 'k'
        cleaned_part = re.sub(r'[^\d.k]', '', part)
        
        if not cleaned_part:
            continue

        num_val = None
        try:
            if 'k' in cleaned_part:
                num_val = float(cleaned_part.replace('k', '')) * 1000
            else:
                num_val = float(cleaned_part)
            numbers.append(num_val)
        except ValueError:
            # Could not convert, skip this part
            pass
            
    # If we parse something like "up to 100k", parts might be ["up ", " 100k"].
    # We need to be more robust. Let's use findall for numbers.
    
    # Regex to find numbers like 50000, 50k, 50.5k
    # It will find all such occurrences.
    potential_salaries = re.findall(r'(\d+\.?\d*k?)', text)
    
    parsed_numbers = []
    for s_val in potential_salaries:
        try:
            if 'k' in s_val:
                num = float(s_val.replace('k', '')) * 1000
            else:
                num = float(s_val)
            # Basic sanity check for salary values (e.g., > 100 assuming not hourly parsed, < 1,000,000)
            if 1000 < num < 1000000: # Heuristic to filter out unlikely numbers
                 parsed_numbers.append(num)
        except ValueError:
            continue
    
    if not parsed_numbers: # If regex findall failed, fallback to simpler split (less reliable)
        return numbers[:2] # Return up to two numbers from split method as a fallback

    return sorted(list(set(parsed_numbers)))[:2] # Return unique, sorted, up to two main salary figures

def process_salary_for_chart(salary_text):
    """
    Processes salary text to get a single representative USD value for plotting.
    If a range is found, it takes the average.
    """
    if pd.isna(salary_text):
        return None

    extracted_salaries_aud = extract_salary_values(salary_text)
    
    if not extracted_salaries_aud:
        return None
        
    avg_salary_aud = sum(extracted_salaries_aud) / len(extracted_salaries_aud)
    
    if avg_salary_aud <= 0: # Invalid salary
        return None

    salary_usd = avg_salary_aud * AUD_TO_USD
    return salary_usd

def create_salary_range_chart(df):
    """Create a box plot for salary ranges by job category converted to USD."""
    category_col = "title_category" if "title_category" in df.columns else "job type"
        
    if "salary" not in df.columns or category_col not in df.columns:
        st.warning(f"Salary chart requires 'salary' and a category column ('title_category' or 'job type').")
        return None
    
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df_copy['salary_value_usd'] = df_copy['salary'].apply(process_salary_for_chart)
    
    salary_df = df_copy.dropna(subset=['salary_value_usd', category_col])
    
    if salary_df.empty:
        st.warning("No valid salary data found after processing for the salary chart.")
        return None
        
    # Consider top N categories by count of jobs with valid salaries
    # Count jobs *after* filtering for valid salary and category
    category_counts = salary_df[category_col].value_counts()
    top_n_categories = category_counts.nlargest(15).index.tolist() # Show up to 15 categories
    
    filtered_df = salary_df[salary_df[category_col].isin(top_n_categories)]
    
    if filtered_df.empty:
        st.warning("No data for top categories after salary processing.")
        return None

    fig = px.box(filtered_df, 
                 x='salary_value_usd', 
                 y=category_col,
                 color=category_col,
                 title=f'Annual Salary Analysis by {category_col.replace("_", " ").title()} (USD)',
                 labels={'salary_value_usd': 'Annual Salary (USD)', category_col: category_col.replace("_", " ").title()},
                 orientation='h',
                 points="outliers")
    
    # Sort y-axis by median salary (descending)
    median_salaries = filtered_df.groupby(category_col)['salary_value_usd'].median().sort_values(ascending=False)
    
    chart_height = max(400, len(median_salaries) * 35 + 150) # Dynamic height

    fig.update_layout(
        showlegend=False,
        xaxis_title='Annual Salary (USD)',
        yaxis_title=category_col.replace("_", " ").title(),
        yaxis={'categoryorder':'array', 'categoryarray': median_salaries.index, 'tickmode': 'array', 'tickvals': median_salaries.index, 'ticktext': [f"{idx} (n={category_counts[idx]})" for idx in median_salaries.index]},
        height=chart_height,
        margin=dict(l=250, r=20, b=50, t=50), # Increased left margin for longer y-axis labels
        template='plotly_white' # Using white template for better contrast with box plots
    )
    return fig

def main():
    """Main function to run the job heatmap dashboard."""
    st.title("🇦🇺 Victoria Job Market Dashboard")
    st.markdown("Explore job postings across Victoria, Australia. Visualize locations, categories, salaries, and more.")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        refresh_interval = st.selectbox(
            "Auto-refresh interval:",
            [None, "1 minute", "5 minutes", "10 minutes", "30 minutes"],
            index=0, key="refresh_interval"
        )
    with col2:
        if st.button("🔄 Refresh Data Now", key="refresh_button"):
            st.cache_data.clear() # Clear all cached functions
            st.success("✅ Cache cleared! Data will be reloaded.")
            # No explicit rerun needed here, Streamlit reruns on widget interaction.
            # Forcing a full rerun if button is pressed AFTER initial load:
            if 'data_loaded_once' in st.session_state and st.session_state.data_loaded_once:
                 st.rerun()


    if refresh_interval and ('data_loaded_once' not in st.session_state or not st.session_state.data_loaded_once): # Avoid rerun on first script run due to this
        interval_map = {"1 minute": 60, "5 minutes": 300, "10 minutes": 600, "30 minutes": 1800}
        # Note: st.experimental_rerun_after is not a standard Streamlit function.
        # If this line causes an error, you'll need an alternative auto-refresh mechanism.
        # For example, using `streamlit-autorefresh` component.
        # st.experimental_rerun_after(interval_map[refresh_interval]) 
        # Commenting out as it's likely non-standard. A true auto-refresh needs a component or JS.
        st.info(f"Auto-refresh set to {refresh_interval}. Manual refresh or component needed for actual auto-trigger.")


    with st.spinner("⏳ Loading and processing job data..."):
        df_loaded = load_data()
    
    st.session_state.data_loaded_once = True


    if df_loaded is not None:
        df = df_loaded.copy() # Work with a copy
        df = categorize_job_titles(df) # This adds 'title_category'
        
        st.success(f"✅ Data Loaded Successfully! Found {len(df)} job postings.")
        
        # Key Metrics Section
        st.subheader("📊 Key Metrics Overview")
        date_col_options = ['date posted', 'posted date', 'date', 'created_at', 'published_date', 'updated_at', 'last_seen']
        actual_date_col = None
        for col_name in date_col_options:
            if col_name in df.columns:
                try:
                    # Coerce errors: invalid dates become NaT
                    temp_dates = pd.to_datetime(df[col_name], errors='coerce')
                    if not temp_dates.isna().all(): # Check if any valid dates were parsed
                        df[col_name] = temp_dates # Assign back if valid
                        actual_date_col = col_name
                        break
                except Exception:
                    continue # Problem with this column, try next

        metric_cols = st.columns(3)
        metric_cols[0].metric("Total Job Postings", len(df))
        metric_cols[1].metric("Unique Locations", df['location'].nunique())
        
        if actual_date_col:
            min_date = df[actual_date_col].min()
            max_date = df[actual_date_col].max()
            date_range_str = "N/A"
            if pd.notna(min_date) and pd.notna(max_date):
                 date_range_str = f"{min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')}"
            metric_cols[2].metric("Postings Date Range", date_range_str)
        else:
            default_cat_col = "title_category" if "title_category" in df.columns else ("job type" if "job type" in df.columns else None)
            if default_cat_col:
                metric_cols[2].metric(f"Unique Categories ({default_cat_col.replace('_', ' ').title()})", df[default_cat_col].nunique())
            else:
                metric_cols[2].metric("Unique Categories", "N/A")


        if st.checkbox("🕵️ Show Raw Data Sample & Schema", key="show_schema"):
            st.write("Available columns:", df.columns.tolist())
            st.dataframe(df.sample(min(5, len(df)))) # Show a random sample
        
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📍 Location Map", "📊 Job Categories", "🔍 Source & Status", "💰 Salary Analysis"])
        
        with tab1:
            st.header("🗺️ Job Locations in Victoria")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            locations_to_geocode = df["location"].unique()
            geocoded_cache = {}
            
            with st.spinner(f"Geocoding {len(locations_to_geocode)} unique locations... (may take time)"):
                for i, loc_name in enumerate(locations_to_geocode):
                    if pd.notna(loc_name):
                        lat, lon = geocode_location(loc_name) # Uses @st.cache_data
                        geocoded_cache[loc_name] = (lat, lon)
                    progress_bar.progress(int((i + 1) / len(locations_to_geocode) * 100))
                    status_text.text(f"Geocoding {i+1}/{len(locations_to_geocode)}: {loc_name}")
                
                df["lat"] = df["location"].map(lambda x: geocoded_cache.get(x, (None, None))[0] if pd.notna(x) else None)
                df["lon"] = df["location"].map(lambda x: geocoded_cache.get(x, (None, None))[1] if pd.notna(x) else None)
                
                valid_geo_data = df.dropna(subset=["lat", "lon"])
                status_text.success(f"🌍 Geocoding complete! {len(valid_geo_data)} of {len(df)} postings have valid coordinates.")
            
            if not valid_geo_data.empty:
                map_options_cols = st.columns(2)
                with map_options_cols[0]:
                    map_type = st.radio("Map Display Type:", ["Heatmap", "Clustered Markers", "Both"], index=1, horizontal=True, key="map_type_select")
                with map_options_cols[1]:
                    tile_layer = st.selectbox("Map Tile Style:", 
                                              ["OpenStreetMap", "CartoDB Positron", "CartoDB DarkMatter", "Stamen Toner", "Stamen Terrain", "Stamen Watercolor"], 
                                              index=1, key="map_tile_select")

                map_center = [-37.8136, 144.9631] # Melbourne
                map_zoom = 7 if len(valid_geo_data) > 10 else 9

                m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=tile_layer)
                folium.plugins.Fullscreen().add_to(m)

                def create_popup_html(row_data):
                    # Gracefully get data, defaulting to "N/A"
                    title = row_data.get("title", row_data.get("job title", "N/A"))
                    company = row_data.get("company", "N/A")
                    loc = row_data.get("location", "N/A")
                    job_type_val = row_data.get("job type", "N/A")
                    title_cat = row_data.get("title_category", "N/A")

                    html = f"""
                    <div style="font-family: Arial, sans-serif; font-size: 13px;">
                    <b>{title}</b><br>
                    🏢 <i>{company}</i><br>
                    📍 {loc}<br>
                    🏷️ Category: {title_cat}<br>
                    💼 Type: {job_type_val}<br>
                    """
                    url_cols = ["url", "link", "joburl", "apply_link", "sourceurl"]
                    job_url = next((row_data[col] for col in url_cols if col in row_data and pd.notna(row_data[col]) and str(row_data[col]).startswith("http")), None)
                    if job_url:
                        html += f'<br><a href="{job_url}" target="_blank" style="color: #007bff; text-decoration: none;">🚀 View Job</a>'
                    html += "</div>"
                    return html

                if map_type in ["Heatmap", "Both"]:
                    from folium.plugins import HeatMap
                    heat_map_data = valid_geo_data[["lat", "lon"]].values.tolist()
                    HeatMap(heat_map_data, radius=12, blur=10, name="Job Density Heatmap").add_to(m)
                
                if map_type in ["Clustered Markers", "Both"]:
                    from folium.plugins import MarkerCluster
                    marker_cluster = MarkerCluster(name="Individual Job Postings").add_to(m)
                    
                    # Icon and color mapping for title_category
                    cat_icon_map = {
                        "Developer": ("cog", "blue"), "Data": ("stats", "green"), "Management": ("briefcase", "purple"),
                        "Design": ("pencil", "orange"), "Marketing": ("bullhorn", "red"), "Finance": ("usd", "darkblue"),
                        "HR": ("users", "cadetblue"), "Support": ("life-ring", "lightred"), "Sales": ("handshake-o", "darkgreen"), # FontAwesome icons
                        "Operations": ("tasks", "beige"), "Consulting": ("comments", "pink"), 
                        "System Admin/DevOps": ("server", "lightblue"), "Quality Assurance": ("check", "lightgreen"),
                        "Security": ("shield", "black"), "Other": ("info-sign", "gray")
                    }

                    for _, row in valid_geo_data.iterrows():
                        popup_content = create_popup_html(row)
                        iframe = folium.IFrame(html=popup_content, width=280, height=150)
                        popup = folium.Popup(iframe, max_width=280)
                        
                        cat = row.get("title_category", "Other")
                        icon_name, color = cat_icon_map.get(cat, cat_icon_map["Other"])
                        
                        # For FontAwesome, prefix is 'fa'
                        # For Glyphicon, prefix is 'glyphicon' - more limited set of free icons
                        # Using glyphicon for broader compatibility without extra CSS/JS unless FontAwesome is explicitly supported by folium easily
                        icon_prefix = 'glyphicon' # Defaulting to glyphicon
                        # Example: If using FontAwesome icons like 'handshake-o'
                        # You might need to ensure FontAwesome is loaded or use a plugin like BeautifyIcon
                        # For simplicity, map to available glyphicons or use generic ones.
                        # Simplified mapping for built-in glyphicons:
                        icon_name_glyphicon = {
                            "cog": "cog", "stats": "stats", "briefcase": "briefcase", "pencil": "pencil", "bullhorn":"bullhorn",
                            "usd": "usd", "users": "user", "life-ring": "support", "handshake-o":"transfer", "tasks":"tasks",
                            "comments":"comment", "server": "hdd", "check":"ok", "shield":"shield", "info-sign":"info-sign"
                        }.get(icon_name, "info-sign")


                        folium.Marker(
                            [row["lat"], row["lon"]],
                            popup=popup,
                            tooltip=f"{row.get('title', 'Job')} at {row['location']}",
                            icon=folium.Icon(color=color, icon=icon_name_glyphicon, prefix=icon_prefix)
                        ).add_to(marker_cluster)

                if map_type == "Both":
                    folium.LayerControl().add_to(m)
                
                # Fit bounds if practical
                if len(valid_geo_data[['lat', 'lon']].drop_duplicates()) > 1: # only if more than 1 unique point
                    m.fit_bounds(valid_geo_data[['lat', 'lon']].values.tolist())

                folium_static(m, width=None, height=650)
                
                st.download_button(
                    "📥 Download Geocoded Data (CSV)",
                    valid_geo_data.to_csv(index=False).encode('utf-8'),
                    "geocoded_job_data.csv", "text/csv", key='download_geo_csv'
                )
            else:
                st.warning("⚠️ No valid geocoded locations to display on the map.")
        
        with tab2:
            st.header("📊 Job Category & Type Analysis")
            
            st.subheader("Job Postings by Main Category (from 'job type' column)")
            job_category_chart_main = create_job_category_chart(df)
            if job_category_chart_main:
                st.altair_chart(job_category_chart_main, use_container_width=True)
            else:
                st.info("ℹ️ 'job type' column not found or no data for this chart.")
                
            st.subheader("Job Categories by Title Keywords (from 'title' column)")
            job_title_category_chart_val = create_job_title_category_chart(df)
            if job_title_category_chart_val:
                st.altair_chart(job_title_category_chart_val, use_container_width=True)
            else:
                st.info("ℹ️ Suitable title column not found or no data for this chart.")
            
            st.subheader("Employment Type Breakdown")
            employment_charts_list = create_employment_type_charts(df)
            if employment_charts_list:
                chart_cols = st.columns(len(employment_charts_list))
                for i, chart_item in enumerate(employment_charts_list):
                    with chart_cols[i]:
                        st.plotly_chart(chart_item, use_container_width=True)
            else:
                st.info("ℹ️ Insufficient data for employment type charts (check 'job type', 'contract type' columns).")
                
        with tab3:
            st.header("🔍 Job Source & Status")
            source_status_cols = st.columns(2)
            with source_status_cols[0]:
                st.subheader("Job Source (Seek vs. Other)")
                job_source_chart_val = create_job_source_chart(df)
                st.plotly_chart(job_source_chart_val, use_container_width=True)
            
            with source_status_cols[1]:
                st.subheader("Job Status (Active/Inactive)")
                job_status_chart_val = create_job_status_chart(df)
                st.plotly_chart(job_status_chart_val, use_container_width=True)
        
        with tab4:
            st.header("💰 Salary Insights (AUD to USD)")
            st.info(f"Note: Salaries are estimates, converted from AUD to USD (1 AUD = {AUD_TO_USD} USD), and primarily based on annual figures. Parsing attempts to handle ranges.")
            
            salary_chart_val = create_salary_range_chart(df) # df should already have 'title_category'
            if salary_chart_val:
                st.plotly_chart(salary_chart_val, use_container_width=True)
            else:
                st.warning("⚠️ Could not generate salary analysis. Ensure 'salary' and a category column ('title_category' or 'job type') exist and contain processable data.")
    else:
        st.error("‼️ No data loaded. Please check the Google Sheet URL and data format.")

if __name__ == "__main__":
    st.set_page_config(page_title="VIC Job Market Visualizer", layout="wide", initial_sidebar_state="auto")
    main()
