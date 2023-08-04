
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import missingno as msno
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x:'%.3f'%x)
pd.set_option('display.max_rows', None)


def load():
    data =pd.read_csv('datasets/Telco-Customer-Churn.csv')
    return data
df=load()


# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)


def check_df(dataframe):
    print("########### Shape #############")
    print(dataframe.shape)
    print("########### Columns ###############")
    print(dataframe.columns)
    print("######### data type ##############")
    print(dataframe.dtypes)
    print("############## Na number #########:")
    print(dataframe.isnull().sum())
    print("######## QUANTILE ##############")
    print(dataframe.quantile([0.00, 0.05, 0.50, 0.95,0.99, 1.00]).T)

check_df(df)
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_col= [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != 'O' and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == 'O' and dataframe[col].nunique() > car_th]
    cat_col= cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    #Num cols
    num_col= [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f'Observations : {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_col, num_col, cat_but_car


cat_col, num_col, cat_but_car = grab_col_names(df)


#####Kategorik değişken analizi


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_col:
    cat_summary(df, col)


#NUMERİK DEĞİŞKEN ANALİZİ:


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_col:
    num_summary(df, col, plot=False)


#NUMERİK DEĞİŞKENLERİN TARGET E GÖRE ANALİZİ

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}, end= "\n\n\n"))

for col in num_col:
    target_summary_with_num(df, "Churn", col)

#KATEGORİK DEĞİŞKENLERİN TARGET E GÖRe ANALİZİ

def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"TARGET_MEAN ": dataframe.groupby(cat_col)[target].mean(),
                        "Count": dataframe[cat_col].value_counts(),
                        "Ratio": 100 *dataframe[cat_col].value_counts() /len(dataframe)}), end="\n\n\n")

#######KORELASYON###################33

df[num_col].corr() #buda bi dataframe gönderir.

# Korelasyon Matrisi

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_col].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.corrwith(df["Churn"]).sort_values(ascending=False)

### FEATURE ENGİNEERİNG

####EKSİK DEĞER ANALİZİ

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns=[col for col in df.columns if df[col].isnull().sum()>0]
    n_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio= (dataframe[na_columns].isnull().sum()/dataframe.shape[0] *100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)


df[df["TotalCharges"].isnull()]["tenure"] #Müşteri hiç kalmadığı için değer boştur bu yüzden yerine 0 koy.
df["TotalCharges"].fillna(0, inplace=True)


##################################
# BASE MODEL KURULUMU
##################################

dff = df.copy()
cat_cols = [col for col in cat_col if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)


y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

# Accuracy: 0.7837
# Recall: 0.6333
# Precision: 0.4843
# F1: 0.5489
# Auc: 0.7282



###############################################

### AYKIRI DEĞER ANALİZİ ####################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3= dataframe[col_name].quantile(q3)
    IQR = quantile3-quantile1
    low_limit = quantile1-1.5*IQR
    upper_limit =quantile3+1.5*IQR
    return low_limit, upper_limit


def check_outlier(dataframe, col_name):
    low_lim, up_lim= outlier_thresholds(dataframe, col_name)
    if dataframe[dataframe[col_name] > up_lim | dataframe[col_name] < low_lim].any(axis= None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    low_lim, up_lim= outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[(dataframe[col_name]<low_lim), col_name]= low_lim
    dataframe.loc[(dataframe[col_name]>up_lim), col_name]=up_lim
for col in num_col:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

################ ÖZELLİK ÇIKARIMI #########
#tenure değikeninden değer oluştur ayı yıla çevir

df.loc[(df["tenure"]>=0 )&(df["tenure"]<=12), "NEW_TENURE_YEAR"]="0-1 Year"
df.loc[(df["tenure"]>12 )&(df["tenure"]<=24), "NEW_TENURE_YEAR"]="1-2 Year"
df.loc[(df["tenure"]>24 )&(df["tenure"]<=36), "NEW_TENURE_YEAR"]="2-3 Year"
df.loc[(df["tenure"]>36)&(df["tenure"]<=48), "NEW_TENURE_YEAR"]="3-4 Year"
df.loc[(df["tenure"]>48 )&(df["tenure"]<=60), "NEW_TENURE_YEAR"]="4-5 Year"
df.loc[(df["tenure"]>60)&(df["tenure"]<=72), "NEW_TENURE_YEAR"]= "5-6 Year"

########### Kontratı 1 veya 2 yıllık müşterileri engaged olarak belirleme

df["NEW_ENGAGED"]= df["Contract"].apply(lambda x: 1 if x not in ["One year", "Two year"] else 0)


# Herhangi bir destek, yedek veya koruma almayan kişiler----
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç müşteri
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)



# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)


# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)



# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)


# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)


#ENCODİNG

cat_cols, num_cols, cat_but_car = grab_col_names(df)
def label_encoder(dataframe, binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_col= [col for col in df.columns if df[col].dtypes =='O' and df[col].nunique()==2]
for col in binary_col:
    df=label_encoder(df, col)

###One hot encoding işlemi
cat_cols = [col for col in cat_cols if col not in binary_col and col not in ["Churn", "NEW_TotalServices"]]

def one_hot_encoding(dataframe, cat_name, drop_first= False):
    dataframe=pd.get_dummies(dataframe, columns=cat_name, drop_first= drop_first)
    return dataframe


one_hot_encoding(df, cat_cols, drop_first=True)


##################################
# MODELLEME
##################################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# nıhaı model
# Accuracy: 0.8
# Recall: 0.66
# Precision: 0.51
# F1: 0.58
# Auc: 0.75

# Base Model
# # Accuracy: 0.7837
# # Recall: 0.6333
# # Precision: 0.4843
# # F1: 0.5489
# # Auc: 0.7282




def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()



