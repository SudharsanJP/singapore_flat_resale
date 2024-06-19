import streamlit as st
import pandas as pd

st.title(':orange[-------singapore flat resale project-------]')

#)dataframe(1) 1990 - 1999
df1 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\singapore flat resale project\1Resale_Flat1990_1999.csv")

#)dataframe(2) 2002 - 2012
df2 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\singapore flat resale project\2Resale_Flat_2000Feb2012.csv")

#)dataframe(3) 2012 - 2014
df3 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\singapore flat resale project\3Resale_Flat_Mar2012toDec2014.csv")

#)dataframe(4) 2017 onwards
df4 = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\singapore flat resale project\4Resale_flat_Jan2017onwards.csv")

#)concatenation of dataframes(1-4)
df = pd.concat([df1, df2, df3,df4], axis=0, ignore_index=True)

#) to check na values in df:
df.isna().sum()

#) to check what unique value is available in the column remaining_lease:
df['remaining_lease'].unique()

#)to get unique value of each columns:
df['month'].unique()

#)town column
df['town'].unique()

#)flat_type column
df['flat_type'].unique()

#)block column
df['block'].unique()

#)street_name column
df['street_name'].unique()

#)storey_range column
df['storey_range'].unique()

#)floor_area_sqm column
df['floor_area_sqm'].unique()

#) flat_model column
df['flat_model'].unique()

#)lease_commence_date column
df['lease_commence_date'].unique()

#)resale_price column
df['resale_price'].unique()

#) filling the null values
temp_df = df.ffill()
temp_df = df.bfill()


#) to check na values in df:
temp_df.isna().sum()

#) encoding the data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['month', 'town', 'flat_type', 'block', 'street_name', 'storey_range',
       'floor_area_sqm', 'flat_model', 'lease_commence_date','remaining_lease']
for col in cols:
    temp_df[col] = le.fit_transform(temp_df[col])

#) 28. button with text
st.subheader("\n:green[1.To know the data below click here]")
if (st.button(':red[click here]')):
    st.markdown("\n#### :blue[1.1 Singapore flat sale sample data:]\n")
    st.dataframe(df.head(4))

    #)after concatenation of df(1-4),counting df
    st.markdown("\n#### :violet[1.2 Total number of data:]\n")
    st.success(len(df))

    #) scatterplot
    import seaborn as sns
    import matplotlib.pyplot as plt
    st.markdown("\n#### :blue[1.3 Scatterplot of data:]\n")
    fig=plt.figure(figsize=(2,2))
    sns.scatterplot(data=temp_df,x ="floor_area_sqm",y='resale_price')  
    st.pyplot(fig)

st.subheader("\n:green[2.Regression ML models]")
selectBox=st.selectbox(":red[**regression models:**]", ['lasso',
                                    'ridge',
                                    'linear'])
                                    
if selectBox == 'lasso':
    #)lasso regression model
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    X = temp_df.drop(['resale_price'],axis=1)
    y = temp_df['resale_price']
    x_train_las,x_test_las,y_train_las,y_test_las = train_test_split(X,y,test_size=0.05)
    
    model = Lasso()
    model.fit(x_train_las,y_train_las)
    train_pred_las = model.predict(x_train_las)
    test_pred_las = model.predict(x_test_las)
    st.markdown("\n##### :violet[2.1.1 Train data(MSE)]")
    st.info(mean_squared_error(y_train_las,train_pred_las)/10000000)
    st.markdown("\n##### :violet[2.1.2 Test data(MSE)]")
    st.info(mean_squared_error(y_test_las,test_pred_las)/10000000)
    
    #) actual testing vs testing prediction
    st.markdown('\n\n##### :violet[2.1.3 Actual testing vs Testing prediction]')
    test_df_las = pd.DataFrame()
    test_df_las['test_actual']= y_test_las
    test_df_las['test_pred'] = test_pred_las
    st.dataframe(test_df_las.head(4))
    
elif selectBox == 'ridge':
    #) Ridge
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X = temp_df.drop(['resale_price'],axis=1)
    y = temp_df['resale_price']
    x_train_rdge,x_test_rdge,y_train_rdge,y_test_rdge = train_test_split(X,y,test_size=0.05)

    model = Ridge()
    model.fit(x_train_rdge,y_train_rdge)
    train_pred_rdge = model.predict(x_train_rdge)
    test_pred_rdge = model.predict(x_test_rdge)
    st.markdown("\n##### :blue[2.1.1 Train data(MSE)]")
    st.success(mean_squared_error(y_train_rdge,train_pred_rdge)/10000000)
    st.markdown("\n##### :blue[2.1.2 Test data(MSE)]")
    st.success(mean_squared_error(y_test_rdge,test_pred_rdge)/10000000)

    #) actual testing vs testing prediction
    st.markdown('\n\n##### :blue[2.1.3 Actual testing vs Testing prediction]')
    test_df_rdge = pd.DataFrame()
    test_df_rdge['test_actual']= y_test_rdge
    test_df_rdge['test_pred'] = test_pred_rdge
    st.dataframe(test_df_rdge.head(4))

elif selectBox == 'linear':
   #) linear
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error
   from sklearn.model_selection import train_test_split

   X = temp_df.drop(['resale_price'],axis=1)
   y = temp_df['resale_price']
   x_train_lnear,x_test_lnear,y_train_lnear,y_test_lnear = train_test_split(X,y,test_size=0.05)
   model = LinearRegression()
   model.fit(x_train_lnear,y_train_lnear)
   train_pred_lnear = model.predict(x_train_lnear)
   test_pred_lnear = model.predict(x_test_lnear)
   
   st.markdown("\n##### :red[3.1.1 Train data(MSE)]")
   st.success(mean_squared_error(y_train_lnear,train_pred_lnear)/10000000)
   st.markdown("\n##### :red[3.1.2 Test data(MSE)]")
   st.success(mean_squared_error(y_test_lnear,test_pred_lnear)/10000000)

   #) actual testing vs testing prediction
   st.markdown('\n##### :red[3.1.3 Actual testing vs Testing prediction]')
   test_df_lnear = pd.DataFrame()
   test_df_lnear['test_actual']= y_test_lnear
   test_df_lnear['test_pred'] = test_pred_lnear
   st.dataframe(test_df_lnear.head(4))

#) ensemble machine learning models
st.subheader('\n:green[3. Classification & Ensemble ML models]')
if (st.checkbox(":violet[**Decision Tree**]")):
    #) DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    
    X = temp_df.drop(['flat_type'],axis=1)
    y = temp_df['flat_type']
    x_train_dcsion,x_test_dcsion,y_train_dcsion,y_test_dcsion = train_test_split(X,y,test_size=0.05)
    
    model = DecisionTreeClassifier()
    model.fit(x_train_dcsion,y_train_dcsion)
    train_pred_dcsion = model.predict(x_train_dcsion)
    test_pred_dcsion = model.predict(x_test_dcsion)
    
    st.markdown('\n##### :red[(i) Train]')
    st.success(f"Accuracy:{accuracy_score(y_train_dcsion,train_pred_dcsion)}")
    
    st.markdown('\n##### :red[(ii) Test]')
    st.success(f"Accuracy: {accuracy_score(y_test_dcsion,test_pred_dcsion)}")
    
    #) actual testing vs testing prediction
    st.markdown('\n##### :red[(iii) Actual testing vs Testing prediction]')
    test_df_dcsion = pd.DataFrame()
    test_df_dcsion['test_actual']= y_test_dcsion
    test_df_dcsion['test_pred'] = test_pred_dcsion
    st.dataframe(test_df_dcsion.head(4))

elif (st.checkbox(":violet[**Random forest**]")):
#) random forest
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    
    X = temp_df.drop(['flat_type'],axis=1)
    y = temp_df['flat_type']
    x_train_rndom,x_test_rndom,y_train_rndom,y_test_rndom = train_test_split(X,y,test_size=0.05)
    
    model = RandomForestClassifier()
    model.fit(x_train_rndom,y_train_rndom)
    train_pred_rndom = model.predict(x_train_rndom)
    test_pred_rndom = model.predict(x_test_rndom)

    st.markdown('\n##### :red[(i) Train]')
    st.success(f"Accuracy: {accuracy_score(y_train_rndom,train_pred_rndom)}")
    st.markdown('\n##### :red[(ii) Test]')
    st.success(f"Accuracy: {accuracy_score(y_test_rndom,test_pred_rndom)}")
    
    #) actual testing vs testing prediction
    st.markdown('\n##### :red[(iii) Actual testing vs Testing prediction]')
    test_df_rndom = pd.DataFrame()
    test_df_rndom['test_actual']= y_test_rndom
    test_df_rndom['test_pred'] = test_pred_rndom
    st.dataframe(test_df_rndom.head(4))

elif (st.checkbox(":violet[**ada**]")):
    #) AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    
    X = temp_df.drop(['flat_type'],axis=1)
    y = temp_df['flat_type']
    x_train_ada,x_test_ada,y_train_ada,y_test_ada = train_test_split(X,y,test_size=0.05)
    
    model = AdaBoostClassifier()
    model.fit(x_train_ada,y_train_ada)
    train_pred_ada = model.predict(x_train_ada)
    test_pred_ada = model.predict(x_test_ada)

    st.markdown('\n##### :red[(i) Train]')
    st.success(f"Accuracy: {accuracy_score(y_train_ada,train_pred_ada)}")
    st.markdown('\n##### :red[(ii) Test]')
    st.success(f"Accuracy: {accuracy_score(y_test_ada,test_pred_ada)}")

    #) actual testing vs testing prediction
    st.markdown('\n##### :red[(iii) Actual testing vs Testing prediction]')
    test_df_ada = pd.DataFrame()
    test_df_ada['test_actual']= y_test_ada
    test_df_ada['test_pred'] = test_pred_ada
    st.dataframe(test_df_ada.head(4))

elif (st.checkbox(":violet[**GBC**]")):
    #) GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    X = temp_df.drop(['flat_type'],axis=1)
    y = temp_df['flat_type']
    x_train_gbc,x_test_gbc,y_train_gbc,y_test_gbc = train_test_split(X,y,test_size=0.05)
    
    model = GradientBoostingClassifier()
    model.fit(x_train_gbc,y_train_gbc)
    train_pred_gbc = model.predict(x_train_gbc)
    test_pred_gbc = model.predict(x_test_gbc)
    
    st.markdown('\n##### :red[(i) Train]')
    st.success(f"Accuracy: {accuracy_score(y_train_gbc,train_pred_gbc)}")
    
    st.markdown('*******Test*******')
    st.success(f"Accuracy: {accuracy_score(y_test_gbc,test_pred_gbc)}")
    
    #) actual testing vs testing prediction
    st.markdown('\n##### :red[(iii) Actual testing vs Testing prediction]')
    test_df_ada = pd.DataFrame()
    test_df_gbc = pd.DataFrame()
    test_df_gbc['test_actual']= y_test_gbc
    test_df_gbc['test_pred'] = test_pred_gbc
    print(test_df_gbc.head(4))

 #) unsupervised learning- KMeans
st.subheader('\n:green[4. Unsupervised learning model - KMeans]')
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(temp_df)
df['class'] = model.labels_
group = df['class'].value_counts()
st.code(f"data grouped into class is{group}")

