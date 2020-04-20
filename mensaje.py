import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
import datetime as dt
import scipy.stats as sp
import datetime

target_date=dt.datetime.today()
# Load the data
#data = pd.read_excel(r'C:\Users\azorjf\Downloads\COVID-19-geographic-disbtribution-worldwide.xlsx')
data = pd.read_excel('https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx')

#dataSP["dateRep"]=pd.to_datetime(dataSP["dateRep"])
#dataSP["dateRep"]=dataSP["dateRep"].map(dt.datetime.toordinal)

#Prepare data
filterSP=data['countriesAndTerritories']=='Spain' #filtro por Spain
start_day=target_date - datetime.timedelta(days=7) #cojo únicamente 7 dias de cara al modelo
filterDateModel=data['dateRep']>=start_day
filterDate=data['dateRep']>='2020-03-13'
dataSP=data[filterSP]
dataSPmodel=dataSP[filterDateModel]
dataSP=dataSP[filterDate]
#dataSP["dateRep"]=pd.to_datetime(dataSP["dateRep"])
#dataSP["Week"]=zip(*pd.to_datetime(dataSP['dateRep'].isocalendar()[1]))
temp=pd.to_datetime(dataSPmodel["dateRep"])
X=np.c_[temp.map(dt.datetime.toordinal)]
y=np.c_[dataSPmodel['cases']]
z=np.c_[dataSPmodel['deaths']]




# Select a linear model
model = sklearn.linear_model.LinearRegression()
modelDeaths = sklearn.linear_model.LinearRegression()
#model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model model.fit( X, y)
model.fit(X,y)
modelDeaths.fit(X,z)

listaY=[]
for dias_prediccion in range (1,7):
    aux=(target_date+datetime.timedelta(days=dias_prediccion)).toordinal()
    listaY.append([datetime.datetime.fromordinal(aux),model.predict([[aux]])[0][0],modelDeaths.predict([[aux]])[0][0]])

dfgraphic=pd.DataFrame(listaY,columns=['dateRep','cases','deaths'])
print(dfgraphic)

frames=[dataSP,dfgraphic]
result=pd.concat(frames,axis=0,join='outer',sort=True)
print(result.head())
#Visualize the data
ax=plt.gca() #get current axis
result.plot(kind='line',x='dateRep',y='cases',ax=ax)
result.plot(kind='line',x='dateRep',y='deaths', color='red',ax=ax)
plt.show()
plt.savefig('output.png')
#print(model.corr())
filename='Covid19'+target_date.strftime("%Y%m%d%H%M%S")
result.to_excel(filename+'.xlsx', index = False)

