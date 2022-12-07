# To run on browser: (base) Joes-MacBook-Pro:~ joetran$ streamlit run OneDrive/Python/Streamlit/cohort.py
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as mpl 
from datetime import date, datetime

import time  # to simulate a real time data, time loop
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development


st.set_page_config(
    page_title="Real-Time Cohorts Dashboard",
    page_icon="📈",
    layout="wide",
)
@st.cache

# Define some functions
def purchase_rate(customer_id):
    purchase_rate = [1]
    counter = 1
    for i in range(1,len(customer_id)):
        if customer_id[i] != customer_id[i-1]:
            purchase_rate.append(1)
            counter = 1
        else:
            counter += 1
            purchase_rate.append(counter)
    return purchase_rate
def join_date(date, purchase_rate):
    join_date = list(range(len(date)))
    for i in range(len(purchase_rate)):
        if purchase_rate[i] == 1:
            join_date[i] = date[i]
        else:
            join_date[i] = join_date[i-1]
    return join_date

def freq_purchase(purchase_rate, day):  # get the frequency (days btw orders)
    freq_purchase = list(range(len(day)))
    for i in range(len(purchase_rate)):
        freq_purchase[i] = 0 if purchase_rate[i] == 1 else (day[i] - day[i-1]).days
        
    return freq_purchase
# or we can just use .diff()/np.timedelta64(1, 'D')


def age_by_month(purchase_rate, month, year, join_month, join_year):
    age_by_month = list(range(len(year)))
    for i in range(len(purchase_rate)):
        if purchase_rate[i] == 1: 
            age_by_month[i] = 0
        else:
            if year[i] == join_year[i]:
                age_by_month[i] = month[i] - join_month[i]
            else:
                age_by_month[i] = month[i] - join_month[i] + 12*(year[i]-join_year[i])
    return age_by_month

def age_by_quarter(purchase_rate, quarter, year, join_quarter, join_year):
    age_by_quarter = list(range(len(year)))
    for i in range(len(purchase_rate)):
        if purchase_rate[i] == 1:
            age_by_quarter[i] = 0
        else:
            if year[i] == join_year[i]:
                age_by_quarter[i] = quarter[i] - join_quarter[i]
            else:
                age_by_quarter[i] = quarter[i] - join_quarter[i] + 4*(year[i]-join_year[i])
    return age_by_quarter

def age_by_year(year, join_year):
    age_by_year = list(range(len(year)))
    for i in range(len(year)):
        age_by_year[i] = year[i] - join_year[i]
    return age_by_year

# get the data
@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv('/Users/joetran/Downloads/sales_2018-01-01_2019-12-31.csv')

df = get_data()

# Process df to get cohorts
@st.experimental_memo
def process_df(df):
    df['month'] = pd.to_datetime(df['day']).dt.month
    df = df[~df['customer_type'].isnull()]
    first_time = df.loc[df['customer_type'] == 'First-time',]
    final = df.loc[df['customer_id'].isin(first_time['customer_id'].values)]
    final = final.drop(columns = ['customer_type'])
    final['day']= pd.to_datetime(final['day'], dayfirst=True)
    sorted_final = final.sort_values(['customer_id','day'])
    sorted_final.reset_index(inplace = True, drop = True)
    april=sorted_final.copy()
    first_time = df.loc[df['customer_type'] == 'First-time',]
    final = df.loc[df['customer_id'].isin(first_time['customer_id'].values)]
    final = final.drop(columns = ['customer_type'])
    final['day']= pd.to_datetime(final['day'], dayfirst=True)
    sorted_final = final.sort_values(['customer_id','day'])
    sorted_final.reset_index(inplace = True, drop = True)
    april=sorted_final.copy()

    april['month'] =pd.to_datetime(april['day']).dt.month
    april['Purchase Rate'] = purchase_rate(april['customer_id'])
    april['Join Date'] = join_date(april['day'], april['Purchase Rate'])
    april['Join Date'] = pd.to_datetime(april['Join Date'], dayfirst=True)
    april['cohort'] = pd.to_datetime(april['Join Date']).dt.strftime('%Y-%m')
    april['year'] = pd.to_datetime(april['day']).dt.year
    april['Join Date Month'] = pd.to_datetime(april['Join Date']).dt.month
    april['Join Date Year'] = pd.to_datetime(april['Join Date']).dt.year
    april['Age by month'] = age_by_month(april['Purchase Rate'], april['month'],april['year'], april['Join Date Month'], april['Join Date Year'])
    return april

april=process_df(df)

# # calculate # NCs per month 
# nc_per_month = np.mean(df.loc[df['customer_type']=='First-time'].groupby(['month']).count()['customer_id']) 
# # calculate RCs per month
# rc_per_month=np.mean(df.loc[df['customer_type']=='Returning'].groupby(['month']).count()['customer_id'])


# cohort by exact numbers
@st.experimental_memo
def cohort_numbers(april):
    april_cohorts = april.groupby(['cohort','Age by month']).nunique()
    april_cohorts = april_cohorts.customer_id.to_frame().reset_index()   # convert series to frame
    april_cohorts = pd.pivot_table(april_cohorts, values = 'customer_id',index = 'cohort', columns= 'Age by month')
    return april_cohorts
april_cohorts = cohort_numbers(april)

def draw_cohorts_table_exact_num(april_cohorts):
    april_cohorts = april_cohorts.astype(str)
    april_cohorts=april_cohorts.replace('nan', '',regex=True)
    return april_cohorts

# cohort by percentage
@st.experimental_memo
def cohort_percent(april_cohorts):
    cohorts = april_cohorts.copy()
    #cohorts = cohorts.replace(np.nan,0,regex=True)
    for i in range(len(cohorts.columns)-1):
        cohorts[i+1] = round(cohorts[i+1]/cohorts[0]*100,2)
    cohorts[0] = cohorts[0]/cohorts[0]
    cohorts['average'] = cohorts.iloc[:,1:-1].mean(axis = 1)   # get the average across all columns
    return cohorts
cohorts = cohort_percent(april_cohorts)
@st.experimental_memo
def draw_cohorts_table_percentage(cohorts):
    for i in range(len(cohorts.columns)-2):
        cohorts[i+1]=cohorts[i+1].apply(lambda x:f"{x}%")
    cohorts = cohorts.astype(str)
    cohorts[0] = "100%"
    cohorts=cohorts.replace('nan%', '',regex=True)
    return cohorts

# cohort by AOV
@st.experimental_memo
def cohort_aov(april):
    april_aov = april.groupby(['cohort','Age by month']).mean().total_sales
    april_aov = april_aov.to_frame().reset_index()
    april_aov['total_sales'] = april_aov['total_sales'].apply(lambda x: round(x,2))
    april_aov =  pd.pivot_table(april_aov,values = 'total_sales', index = 'cohort', columns = 'Age by month')    
    return april_aov

april_aov = cohort_aov(april)
def draw_cohorts_aov(april_aov):
    april_aov = april_aov.astype(str)
    april_aov = april_aov.replace('nan', '',regex=True)
    return april_aov


st.markdown("""
This webapp performs cohort analysis of my_company data!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Shopify](https://company_name.myshopify.com/admin).
""")

# dashboard title
st.title(f"Live {cohorts.index[0]} to {cohorts.index[-1]} Cohort Dashboard")

# top-level filters
cohort_tabletype_filter = st.selectbox('Select type of cohort',['By unique customers', 'By percentage', 'By AOV'])

cohort_filter = st.multiselect('Select cohort', list(cohorts.index))
# dataframe filter
@st.experimental_memo
def select_which_table_to_draw(cohort_tabletype_filter,cohort_filter):
    if cohort_tabletype_filter=='By unique customers':
        result = draw_cohorts_table_exact_num(april_cohorts)
        if cohort_filter != []:
            return result.loc[cohort_filter,:]
        else: return result
    elif cohort_tabletype_filter=='By percentage':
        result = draw_cohorts_table_percentage(cohorts)
        if cohort_filter != []:
            return result.loc[cohort_filter,:]
        else: return result
    elif cohort_tabletype_filter=='By AOV':
        result = draw_cohorts_aov(april_aov)
        if cohort_filter != []:
            return result.loc[cohort_filter,:]
        else: 
            return result

draw_cohorts = lambda x,y: select_which_table_to_draw(x,y)
#st.dataframe(draw_cohorts.loc[cohort_filter,:] if cohort_filter !=[] else draw_cohorts)
output = draw_cohorts(cohort_tabletype_filter,cohort_filter)
st.dataframe(output)
    
st.download_button(label='Download csv', data=output.to_csv(), mime='text/csv')
    
# create three columns
kpi1, kpi2, kpi3 = st.columns(3)

# fill in those three columns with respective metrics or KPIs

aov = np.mean(df['total_sales'])
aov_goal = 95.00
nc = np.mean(df.loc[df['customer_type']=='First-time'].groupby(['day']).count()['customer_id'])
nc_goal = 30
rc = np.mean(df.loc[df['customer_type']=='Returning'].groupby(['day']).count()['customer_id'])
rc_goal = 250


kpi1.metric(
    label="AOV",
    value=f"$ {round(aov,2)}",
    delta=f"-${round(aov_goal-aov,2)}" if aov_goal>aov else f"${round(aov-aov_goal,2)}",
)

kpi2.metric(
    label="New customers/day",
    value=int(nc),
    delta=f"-{round((nc_goal-nc)/nc_goal*100,2)}%" if nc_goal>nc else f"{round((nc - nc_goal)/nc_goal*100,0)}%",
)

kpi3.metric(
    label="Returning customers/day",
    value= int(rc),
    delta=f"-{round((rc_goal - rc)/rc_goal*100,2)}%" if rc_goal>rc else f"{round((rc-rc_goal)/rc_goal*100,2)}%"
)

# Interactive charts
# fig_col1, fig_col2 = st.columns(2)

cohorts_t = cohorts.transpose()


# sns.set(style='whitegrid')
# plt.figure(figsize=(20, 15))
# plt.title('Cohorts: User Retention')
# sns.set(font_scale = 1) # font size

# sns.heatmap(cohorts, mask=cohorts.isnull(), 
#                     cmap="Greens",
#                     annot=True, fmt='.01%')
# plt.show()




# with fig_col1:
# fig1 = plt.figure() 
# plt.plot([1, 2, 3, 4, 5]) 
# plt.show()
# plt.title('Cohorts: User Retention')
# plt.figure(figsize=(20, 15))
# st.pyplot(fig1)

# sns.set(style='whitegrid')
# fig2 = plt.figure()
# plt.plot(cohorts_t[:,:-1])

# st.pyplot(fig2)

# with fig_col2:
#     st.markdown("### Second Chart")
#     fig2 = px.histogram(data_frame=df, x="age_new")
#     st.write(fig2)

