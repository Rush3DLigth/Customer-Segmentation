import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#from dataprep.eda import plot_correlation
from datetime import timedelta
import matplotlib.pyplot as plt

# Web App Title
st.markdown('''
# **The Customer Segmentation RFM model using K-Means algorithms App**

---
''')

# Upload excel data
with st.sidebar.header('Upload your excel data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input excel file", type=["xlsx"])

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_excel():
        excel = pd.read_excel(uploaded_file) ##excel file
        return excel
    df = load_excel()

    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')

    #df['Price'] = df['Quantity'] * df['UnitPrice']
    #order = df.groupby(['InvoiceNo','InvoiceDate','CustomerID']).agg({'Price':lambda x:x.sum()}).reset_index()
   
    st.header('**Checking Dataframe**')
    df_shape = df.shape
    st.write('Shape : ', df_shape)
    #column data type
    df_data_type = df.dtypes
    st.header('Data Types')
    st.write(df_data_type)
    #Number of NaN values per column:
    df_Number_NaN = df.isnull().sum()
    st.write('Number of NaN values per column : ',df_Number_NaN)
    #st.write(order)
    st.write('---')
    st.header('**Statistics**')
    df_describe = df.describe().round(2)
    st.write(df_describe)

    st.header('**Group by Invoice No.**')
    df['Price'] = df['Quantity'] * df['UnitPrice']
    df_order = df.groupby(['InvoiceNo','InvoiceDate','CustomerID']).agg({'Price':lambda x:x.sum()}).reset_index()
    st.write(df_order.head())
    st.write(df_order.describe())
    NOW = df_order['InvoiceDate'].max() + timedelta(days=1)
    period = 365
    df_order['Recency'] = df_order['InvoiceDate'].apply(lambda x:(NOW - x).days)
    aggr = {'Recency':lambda x:x.min(), #the number of days since last order (Recency)
    'InvoiceDate':lambda x:len([d for d in x if d >= NOW - timedelta(days=period)]),} # the total number of order in the last period (Frequency)
    rfm = df_order.groupby('CustomerID').agg(aggr).reset_index()
    rfm.rename(columns={'InvoiceDate':'Frequency'},inplace=True)
    rfm['Monetary'] = rfm['CustomerID'].apply(lambda x:df_order[(df_order['CustomerID']==x) & (df_order['InvoiceDate'] >= NOW - timedelta(days=period))]['Price'].sum())
    st.header('**RFM Table**')
    st.write(rfm.head())
    st.header('**RFM Statistics**')
    st.write(rfm.describe())

    #Assige score to RFM
    quintiles = rfm[['Recency','Frequency','Monetary']].quantile([.2, .4, .6, .8]).to_dict() #ควอนไทล์แจกแจง
    def r_score(x):
        if x <= quintiles['Recency'][.2]:
            return 5
        elif x <= quintiles['Recency'][.4]:
            return 4
        elif x <= quintiles['Recency'][.6]:
            return 3
        elif x <= quintiles['Recency'][.8]:
            return 2
        else:
            return 1
    def fm_score(x, c):
        if x <= quintiles[c][.2]:
            return 1
        if x <= quintiles[c][.4]:
            return 2
        if x <= quintiles[c][.6]:
            return 3
        if x <= quintiles[c][.8]:
            return 4
        else:
            return 5

    rfm['R'] = rfm['Recency'].apply(lambda x : r_score(x))
    rfm['F'] = rfm['Frequency'].apply(lambda x : fm_score(x, 'Frequency'))
    rfm['M'] = rfm['Monetary'].apply(lambda x : fm_score(x, 'Monetary'))

    rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
    st.header('**RFM Score Table**')
    st.write(rfm)

    segt_map = {
        r'[1-2][1-2]': 'Lost',
        r'[1-2][3-4]': 'Sleeper',
        r'[1-2]5': 'Shouldn\'t Lose',
        r'3[1-2]': 'Cold Leads',
        r'33': 'need attention',
        r'[3-4][4-5]': 'loyal customers',
        r'41': 'Warm Leads',
        r'51': 'new customers',
        r'[4-5][2-3]': 'hopeful',
        r'5[4-5]': 'champions'
    }

    rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
    rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)
    ##-----Table------##
    st.header('**Segmentation Table**')
    st.write(rfm)

    # plot the distribution of customers over R and F
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    for i, p in enumerate(['R', 'F']):
        parameters = {'R':'Recency', 'F':'Frequency'}
        y = rfm[p].value_counts().sort_index()
        x = y.index
        ax = axes[i]
        bars = ax.bar(x, y, color='silver')
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_title('Distribution of {}'.format(parameters[p]),
                    fontsize=14)
        for bar in bars:
            value = bar.get_height()
            if value == y.max():
                bar.set_color('firebrick')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value - 5,
                    '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
                ha='center',
                va='top',
                color='w')

    #plt.show()
    plt.tight_layout()
    st.write(fig)

    # plot the distribution of M for RF score
    fig, axes = plt.subplots(nrows=5, ncols=5,
                            sharex=False, sharey=True,
                            figsize=(10, 10))

    r_range = range(1, 6)
    f_range = range(1, 6)
    for r in r_range:
        for f in f_range:
            y = rfm[(rfm['R'] == r) & (rfm['F'] == f)]['M'].value_counts().sort_index()
            x = y.index
            ax = axes[r - 1, f - 1]
            bars = ax.bar(x, y, color='silver')
            if r == 5:
                if f == 3:
                    ax.set_xlabel('{}\nF'.format(f), va='top')
                else:
                    ax.set_xlabel('{}\n'.format(f), va='top')
            if f == 1:
                if r == 3:
                    ax.set_ylabel('R\n{}'.format(r))
                else:
                    ax.set_ylabel(r)
            ax.set_frame_on(False)
            ax.tick_params(left=False, labelleft=False, bottom=False)
            ax.set_xticks(x)
            ax.set_xticklabels(x, fontsize=8)

            for bar in bars:
                value = bar.get_height()
                if value == y.max():
                    bar.set_color('firebrick')
                ax.text(bar.get_x() + bar.get_width() / 2,
                        value,
                        int(value),
                        ha='center',
                        va='bottom',
                        color='k')
    fig.suptitle('Distribution of M for each F and R',
                fontsize=14)
    #plt.tight_layout()
    #st.write(fig)



    # count the number of customers in each segment
    segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots()

    bars = ax.barh(range(len(segments_counts)),
                segments_counts,
                color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
                bottom=False,
                labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)

    for i, bar in enumerate(bars):
            value = bar.get_width()
            if segments_counts.index[i] in ['champions', 'loyal customers']:
                bar.set_color('firebrick')
            ax.text(value,
                    bar.get_y() + bar.get_height()/2,
                    '{:,} ({:}%)'.format(int(value),
                                    int(value*100/segments_counts.sum())),
                    va='center',
                    ha='left'
                )
    plt.tight_layout()
    st.write(fig)

    st.write('---') 
    st.header('**Apply K-means**')

    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for excel file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Profiling Report**')
        st_profile_report(pr)
