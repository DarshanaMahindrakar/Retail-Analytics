############ PRICE OPTIMIZATION FOR RETAIL PRODUCTS ###############

# Importing data From Postgresql

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import psycopg2
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt




conn = psycopg2.connect(dbname='priceoptimize', user='postgres', password='Post@123', host='localhost', port='5432')
cur = conn.cursor()
curs = conn.cursor()
curs.execute("ROLLBACK")
conn.commit()
cur.execute('SELECT * FROM public."DATA"')

#cur.execute('SELECT * FROM data optimize ORDER BY zone, name, brand, mc')
df = cur.fetchall()

# Creating a DataFrame
df1 = pd.DataFrame(df)

df1 = df1.rename({0: 'UID'}, axis=1)
df1 = df1.rename({1: 'NAME'}, axis=1)
df1 = df1.rename({2: 'ZONE'}, axis=1)
df1 = df1.rename({3: 'Brand'}, axis=1)
df1 = df1.rename({4: 'MC'}, axis=1)
df1 = df1.rename({5: 'Fdate'}, axis=1)
df1 = df1.rename({6: 'quantity'}, axis=1)
df1 = df1.rename({7: 'NSV'}, axis=1)
df1 = df1.rename({8: 'GST_Value'}, axis=1)
df1 = df1.rename({9: 'NSV-GST'}, axis=1)
df1 = df1.rename({10: 'sales_at _cost'}, axis=1)
df1 = df1.rename({11: 'SALES_AT_COST'}, axis=1)
df1 = df1.rename({12: 'MARGIN%'}, axis=1)
df1 = df1.rename({13: 'Gross_Sales'}, axis=1)
df1 = df1.rename({14: 'GrossRGM(P-L)'}, axis=1)
df1 = df1.rename({15: 'Gross_ Margin%(Q/P*100)'}, axis=1)
df1 = df1.rename({16: 'MRP'}, axis=1)
df1 = df1.rename({17: 'price'}, axis=1)
df1 = df1.rename({18: 'DIS'}, axis=1)
df1 = df1.rename({19: 'DIS%'}, axis=1)
df1[['quantity', 'NSV', 'GST_Value', 'NSV-GST', 'sales_at _cost', 'SALES_AT_COST', 'MARGIN%', 'Gross_Sales', 'GrossRGM(P-L)', 'Gross_ Margin%(Q/P*100)', 'MRP', 'price', 'DIS', 'DIS%']] = df1[['quantity', 'NSV', 'GST_Value', 'NSV-GST', 'sales_at _cost', 'SALES_AT_COST', 'MARGIN%', 'Gross_Sales', 'GrossRGM(P-L)', 'Gross_ Margin%(Q/P*100)', 'MRP', 'price', 'DIS', 'DIS%']].apply(pd.to_numeric)

data = df1.drop_duplicates()


st.title('Price Optimization')

Unique_Products = pickle.load(open('Unique_Products.pkl','rb'))
Zone = pickle.load(open('Zone.pkl','rb'))



Selected_Product_Name = st.selectbox(
    'Select Product Name',
     (Unique_Products.values))

Selected_Zone = st.selectbox(
    'Select Zone',
     (Zone.values))

data = data.loc[data['NAME'] == Selected_Product_Name,:]
data_new = data.loc[data['ZONE'] == Selected_Zone,:]
values_at_max_profit = 0


def find_optimal_price(data_new):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    # demand curve
    sns.lmplot(x="price", y="quantity", data=data_new, fit_reg=True, size=4)
    # fit OLS model
    model = ols("quantity ~ price", data=data_new).fit()
    # print model summary
    print(model.summary())

    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_partregress_grid(model, fig=fig)

    fig = plt.figure(figsize=(12, 8))
    fig = sm.graphics.plot_regress_exog(model, "price", fig=fig)

    prams = model.params
    #prams.Intercept
    #prams.price

    # plugging regression coefficients
    # quantity = prams.Intercept + prams.price * price # eq (5)
    # the profit function in eq (3) becomes
    # profit = (prams.Intercept + prams.price * price) * price - cost # eq (6)

    # a range of diffferent prices to find the optimum one
    start_price = data_new.price.min()
    end_price = data_new.price.max()
    Price = np.arange(start_price, end_price, 0.05)
    Price = list(Price)

    # assuming a fixed cost
    k1 = data_new['sales_at _cost'].div(data_new['quantity'])
    cost = k1.min()
    Profit = []
    Quantity = []
    for i in Price:
        GST = 0.05 * i
        quantity_demanded = prams.Intercept + prams.price * i
        Quantity.append(quantity_demanded)

        # profit function
        Profit.append((i - cost - GST) * quantity_demanded)
    # create data frame of price and revenue
    frame = pd.DataFrame({"Price": Price, 'Quantity': Quantity, "Profit": Profit})

    # plot revenue against price
    plt.plot(frame["Price"], frame["Profit"])

    # price at which revenue is maximum

    ind = np.where(frame['Profit'] == frame['Profit'].max())[0][0]
    values_at_max_profit = frame.iloc[[ind]]
    return values_at_max_profit



#optimal_price = {}
#optimal_price[Selected_Product_Name] = find_optimal_price(data_new)
#optimal_price[Selected_Product_Name]

if st.button('Predict Optimized Price'):
    values_at_max_profit = find_optimal_price(data_new)
    st.write('Optimized Price of the Product', Selected_Product_Name, values_at_max_profit)
