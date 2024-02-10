---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.7
  nbformat: 4
  nbformat_minor: 5
---

::: {#962cf961 .cell .code execution_count="491"}
``` python
#import all requrired libraries
import numpy as np
import pandas as pd
```
:::

::: {#e34e5b9b .cell .code execution_count="492"}
``` python
#import data
data=pd.read_csv("Bengaluru_House_Data.csv")
```
:::

::: {#4dd8ef8d .cell .code execution_count="493"}
``` python
data.head()
```

::: {.output .execute_result execution_count="493"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area_type</th>
      <th>availability</th>
      <th>location</th>
      <th>size</th>
      <th>society</th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>balcony</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Super built-up  Area</td>
      <td>19-Dec</td>
      <td>Electronic City Phase II</td>
      <td>2 BHK</td>
      <td>Coomee</td>
      <td>1056</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>39.07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Plot  Area</td>
      <td>Ready To Move</td>
      <td>Chikka Tirupathi</td>
      <td>4 Bedroom</td>
      <td>Theanmp</td>
      <td>2600</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>120.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Built-up  Area</td>
      <td>Ready To Move</td>
      <td>Uttarahalli</td>
      <td>3 BHK</td>
      <td>NaN</td>
      <td>1440</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>62.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Super built-up  Area</td>
      <td>Ready To Move</td>
      <td>Lingadheeranahalli</td>
      <td>3 BHK</td>
      <td>Soiewre</td>
      <td>1521</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>95.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Super built-up  Area</td>
      <td>Ready To Move</td>
      <td>Kothanur</td>
      <td>2 BHK</td>
      <td>NaN</td>
      <td>1200</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>51.00</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#0aa303f6 .cell .code execution_count="494"}
``` python
data.shape
```

::: {.output .execute_result execution_count="494"}
    (13320, 9)
:::
:::

::: {#c521c2b0 .cell .code execution_count="495"}
``` python
data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13320 entries, 0 to 13319
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   area_type     13320 non-null  object 
     1   availability  13320 non-null  object 
     2   location      13319 non-null  object 
     3   size          13304 non-null  object 
     4   society       7818 non-null   object 
     5   total_sqft    13320 non-null  object 
     6   bath          13247 non-null  float64
     7   balcony       12711 non-null  float64
     8   price         13320 non-null  float64
    dtypes: float64(3), object(6)
    memory usage: 936.7+ KB
:::
:::

::: {#35408ad6 .cell .code execution_count="496"}
``` python
data.columns
```

::: {.output .execute_result execution_count="496"}
    Index(['area_type', 'availability', 'location', 'size', 'society',
           'total_sqft', 'bath', 'balcony', 'price'],
          dtype='object')
:::
:::

::: {#bf698391 .cell .code execution_count="497"}
``` python
#value count for each column 
for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)
```

::: {.output .stream .stdout}
    Super built-up  Area    8790
    Built-up  Area          2418
    Plot  Area              2025
    Carpet  Area              87
    Name: area_type, dtype: int64
    ********************
    Ready To Move    10581
    18-Dec             307
    18-May             295
    18-Apr             271
    18-Aug             200
                     ...  
    15-Aug               1
    17-Jan               1
    16-Nov               1
    16-Jan               1
    14-Jul               1
    Name: availability, Length: 81, dtype: int64
    ********************
    Whitefield                        540
    Sarjapur  Road                    399
    Electronic City                   302
    Kanakpura Road                    273
    Thanisandra                       234
                                     ... 
    Bapuji Layout                       1
    1st Stage Radha Krishna Layout      1
    BEML Layout 5th stage               1
    singapura paradise                  1
    Abshot Layout                       1
    Name: location, Length: 1305, dtype: int64
    ********************
    2 BHK         5199
    3 BHK         4310
    4 Bedroom      826
    4 BHK          591
    3 Bedroom      547
    1 BHK          538
    2 Bedroom      329
    5 Bedroom      297
    6 Bedroom      191
    1 Bedroom      105
    8 Bedroom       84
    7 Bedroom       83
    5 BHK           59
    9 Bedroom       46
    6 BHK           30
    7 BHK           17
    1 RK            13
    10 Bedroom      12
    9 BHK            8
    8 BHK            5
    11 BHK           2
    11 Bedroom       2
    10 BHK           2
    14 BHK           1
    13 BHK           1
    12 Bedroom       1
    27 BHK           1
    43 Bedroom       1
    16 BHK           1
    19 BHK           1
    18 Bedroom       1
    Name: size, dtype: int64
    ********************
    GrrvaGr    80
    PrarePa    76
    Sryalan    59
    Prtates    59
    GMown E    56
               ..
    Amionce     1
    JaghtDe     1
    Jauraht     1
    Brity U     1
    RSntsAp     1
    Name: society, Length: 2688, dtype: int64
    ********************
    1200    843
    1100    221
    1500    205
    2400    196
    600     180
           ... 
    3580      1
    2461      1
    1437      1
    2155      1
    4689      1
    Name: total_sqft, Length: 2117, dtype: int64
    ********************
    2.0     6908
    3.0     3286
    4.0     1226
    1.0      788
    5.0      524
    6.0      273
    7.0      102
    8.0       64
    9.0       43
    10.0      13
    12.0       7
    13.0       3
    11.0       3
    16.0       2
    27.0       1
    40.0       1
    15.0       1
    14.0       1
    18.0       1
    Name: bath, dtype: int64
    ********************
    2.0    5113
    1.0    4897
    3.0    1672
    0.0    1029
    Name: balcony, dtype: int64
    ********************
    75.00     310
    65.00     302
    55.00     275
    60.00     270
    45.00     240
             ... 
    351.00      1
    54.10       1
    80.64       1
    32.73       1
    488.00      1
    Name: price, Length: 1994, dtype: int64
    ********************
:::
:::

::: {#51e4a10c .cell .code execution_count="498"}
``` python
# checking null values
data.isna().sum()
```

::: {.output .execute_result execution_count="498"}
    area_type          0
    availability       0
    location           1
    size              16
    society         5502
    total_sqft         0
    bath              73
    balcony          609
    price              0
    dtype: int64
:::
:::

::: {#d3c920fc .cell .code execution_count="499"}
``` python
#dropping some columns we are not much relevant
data.drop(columns=['area_type','availability','society','balcony'],inplace=True)
```
:::

::: {#e2c33b9d .cell .code execution_count="500"}
``` python
data.describe()
```

::: {.output .execute_result execution_count="500"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bath</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13247.000000</td>
      <td>13320.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.692610</td>
      <td>112.565627</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.341458</td>
      <td>148.971674</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40.000000</td>
      <td>3600.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#37309d15 .cell .code execution_count="501"}
``` python
#find all null values and fill them with most occuring value
data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13320 entries, 0 to 13319
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   location    13319 non-null  object 
     1   size        13304 non-null  object 
     2   total_sqft  13320 non-null  object 
     3   bath        13247 non-null  float64
     4   price       13320 non-null  float64
    dtypes: float64(2), object(3)
    memory usage: 520.4+ KB
:::
:::

::: {#e00f4a8b .cell .code execution_count="502"}
``` python
data['location'].value_counts()
```

::: {.output .execute_result execution_count="502"}
    Whitefield                        540
    Sarjapur  Road                    399
    Electronic City                   302
    Kanakpura Road                    273
    Thanisandra                       234
                                     ... 
    Bapuji Layout                       1
    1st Stage Radha Krishna Layout      1
    BEML Layout 5th stage               1
    singapura paradise                  1
    Abshot Layout                       1
    Name: location, Length: 1305, dtype: int64
:::
:::

::: {#6120a5bd .cell .code execution_count="503"}
``` python
#fill null column
data['location']=data['location'].fillna('Sarjapur Road')
```
:::

::: {#2dbe10f8 .cell .code execution_count="504"}
``` python
data['size'].value_counts()
```

::: {.output .execute_result execution_count="504"}
    2 BHK         5199
    3 BHK         4310
    4 Bedroom      826
    4 BHK          591
    3 Bedroom      547
    1 BHK          538
    2 Bedroom      329
    5 Bedroom      297
    6 Bedroom      191
    1 Bedroom      105
    8 Bedroom       84
    7 Bedroom       83
    5 BHK           59
    9 Bedroom       46
    6 BHK           30
    7 BHK           17
    1 RK            13
    10 Bedroom      12
    9 BHK            8
    8 BHK            5
    11 BHK           2
    11 Bedroom       2
    10 BHK           2
    14 BHK           1
    13 BHK           1
    12 Bedroom       1
    27 BHK           1
    43 Bedroom       1
    16 BHK           1
    19 BHK           1
    18 Bedroom       1
    Name: size, dtype: int64
:::
:::

::: {#6c17fc9b .cell .code execution_count="505"}
``` python
data['size']=data['size'].fillna('2 BHK')
```
:::

::: {#a565a479 .cell .code execution_count="506"}
``` python
data['bath']=data['bath'].fillna(data['bath'].median())
```
:::

::: {#6a7515b1 .cell .code execution_count="507"}
``` python
data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13320 entries, 0 to 13319
    Data columns (total 5 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   location    13320 non-null  object 
     1   size        13320 non-null  object 
     2   total_sqft  13320 non-null  object 
     3   bath        13320 non-null  float64
     4   price       13320 non-null  float64
    dtypes: float64(2), object(3)
    memory usage: 520.4+ KB
:::
:::

::: {#1448243d .cell .code execution_count="508"}
``` python
data['size'].str.split()
```

::: {.output .execute_result execution_count="508"}
    0            [2, BHK]
    1        [4, Bedroom]
    2            [3, BHK]
    3            [3, BHK]
    4            [2, BHK]
                 ...     
    13315    [5, Bedroom]
    13316        [4, BHK]
    13317        [2, BHK]
    13318        [4, BHK]
    13319        [1, BHK]
    Name: size, Length: 13320, dtype: object
:::
:::

::: {#6a8f77c1 .cell .code execution_count="509"}
``` python
data['size'].str.split().str.get(0)
```

::: {.output .execute_result execution_count="509"}
    0        2
    1        4
    2        3
    3        3
    4        2
            ..
    13315    5
    13316    4
    13317    2
    13318    4
    13319    1
    Name: size, Length: 13320, dtype: object
:::
:::

::: {#215e5253 .cell .code execution_count="510"}
``` python
#store value of no.of bedrooms to bhk column
data['bhk']=data['size'].str.split().str.get(0).astype(int)
```
:::

::: {#2e35ac7c .cell .code execution_count="511"}
``` python
data[data.bhk>20]
```

::: {.output .execute_result execution_count="511"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>size</th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1718</th>
      <td>2Electronic City Phase II</td>
      <td>27 BHK</td>
      <td>8000</td>
      <td>27.0</td>
      <td>230.0</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4684</th>
      <td>Munnekollal</td>
      <td>43 Bedroom</td>
      <td>2400</td>
      <td>40.0</td>
      <td>660.0</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#e09402ad .cell .code execution_count="512"}
``` python
#search for range values and will take mean of two value
data['total_sqft'].unique()
```

::: {.output .execute_result execution_count="512"}
    array(['1056', '2600', '1440', ..., '1133 - 1384', '774', '4689'],
          dtype=object)
:::
:::

::: {#8720e0b4 .cell .code execution_count="513"}
``` python
data['total_sqft']
```

::: {.output .execute_result execution_count="513"}
    0        1056
    1        2600
    2        1440
    3        1521
    4        1200
             ... 
    13315    3453
    13316    3600
    13317    1141
    13318    4689
    13319     550
    Name: total_sqft, Length: 13320, dtype: object
:::
:::

::: {#8c95b460 .cell .code execution_count="514"}
``` python
#mean
def convertRange(x):
    
    temp=x.split('-')
    if len(temp)==2:
        return(float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
```
:::

::: {#b73238f9 .cell .code execution_count="515"}
``` python
data['total_sqft']=data['total_sqft'].apply(convertRange)
```
:::

::: {#8ab1aad2 .cell .code execution_count="516"}
``` python
data.head()
```

::: {.output .execute_result execution_count="516"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>size</th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Electronic City Phase II</td>
      <td>2 BHK</td>
      <td>1056.0</td>
      <td>2.0</td>
      <td>39.07</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chikka Tirupathi</td>
      <td>4 Bedroom</td>
      <td>2600.0</td>
      <td>5.0</td>
      <td>120.00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Uttarahalli</td>
      <td>3 BHK</td>
      <td>1440.0</td>
      <td>2.0</td>
      <td>62.00</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lingadheeranahalli</td>
      <td>3 BHK</td>
      <td>1521.0</td>
      <td>3.0</td>
      <td>95.00</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kothanur</td>
      <td>2 BHK</td>
      <td>1200.0</td>
      <td>2.0</td>
      <td>51.00</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#3719e308 .cell .code execution_count="517"}
``` python
data['price_per_sqft']=data['price']*100000/data['total_sqft']
```
:::

::: {#95588f4a .cell .code execution_count="518"}
``` python
data['price_per_sqft']
```

::: {.output .execute_result execution_count="518"}
    0         3699.810606
    1         4615.384615
    2         4305.555556
    3         6245.890861
    4         4250.000000
                 ...     
    13315     6689.834926
    13316    11111.111111
    13317     5258.545136
    13318    10407.336319
    13319     3090.909091
    Name: price_per_sqft, Length: 13320, dtype: float64
:::
:::

::: {#f850eea7 .cell .code execution_count="519"}
``` python
data.describe()
```

::: {.output .execute_result execution_count="519"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
      <th>price_per_sqft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13274.000000</td>
      <td>13320.000000</td>
      <td>13320.000000</td>
      <td>13320.000000</td>
      <td>1.327400e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1559.626694</td>
      <td>2.688814</td>
      <td>112.565627</td>
      <td>2.802778</td>
      <td>7.907501e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1238.405258</td>
      <td>1.338754</td>
      <td>148.971674</td>
      <td>1.294496</td>
      <td>1.064296e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>2.678298e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1100.000000</td>
      <td>2.000000</td>
      <td>50.000000</td>
      <td>2.000000</td>
      <td>4.266865e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1276.000000</td>
      <td>2.000000</td>
      <td>72.000000</td>
      <td>3.000000</td>
      <td>5.434306e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1680.000000</td>
      <td>3.000000</td>
      <td>120.000000</td>
      <td>3.000000</td>
      <td>7.311746e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>52272.000000</td>
      <td>40.000000</td>
      <td>3600.000000</td>
      <td>43.000000</td>
      <td>1.200000e+07</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#ee837276 .cell .code execution_count="520"}
``` python
data['location'].value_counts()
```

::: {.output .execute_result execution_count="520"}
    Whitefield                        540
    Sarjapur  Road                    399
    Electronic City                   302
    Kanakpura Road                    273
    Thanisandra                       234
                                     ... 
    1st Stage Radha Krishna Layout      1
    BEML Layout 5th stage               1
    singapura paradise                  1
    Uvce Layout                         1
    Abshot Layout                       1
    Name: location, Length: 1306, dtype: int64
:::
:::

::: {#daa69bef .cell .code execution_count="521"}
``` python
#remove whitespaces from both sides
data['location']=data['location'].apply(lambda x:x.strip())
location_count=data['location'].value_counts()
```
:::

::: {#fcda5641 .cell .code execution_count="522"}
``` python
data['location'].value_counts()
```

::: {.output .execute_result execution_count="522"}
    Whitefield                            541
    Sarjapur  Road                        399
    Electronic City                       304
    Kanakpura Road                        273
    Thanisandra                           237
                                         ... 
    1Channasandra                           1
    Hosahalli                               1
    Vijayabank bank layout                  1
    near Ramanashree California resort      1
    Abshot Layout                           1
    Name: location, Length: 1295, dtype: int64
:::
:::

::: {#590b0e33 .cell .code execution_count="523"}
``` python
location_count_less_10=location_count[location_count<=10]
location_count_less_10
```

::: {.output .execute_result execution_count="523"}
    BTM 1st Stage                         10
    Nagadevanahalli                       10
    Basapura                              10
    Sector 1 HSR Layout                   10
    Dairy Circle                          10
                                          ..
    1Channasandra                          1
    Hosahalli                              1
    Vijayabank bank layout                 1
    near Ramanashree California resort     1
    Abshot Layout                          1
    Name: location, Length: 1054, dtype: int64
:::
:::

::: {#9e11fe97 .cell .code execution_count="524"}
``` python
#replace location with other having location count less than 0
data['location']=data['location'].apply(lambda x:'other' if x in location_count_less_10 else x)
data['location'].value_counts()
```

::: {.output .execute_result execution_count="524"}
    other                 2886
    Whitefield             541
    Sarjapur  Road         399
    Electronic City        304
    Kanakpura Road         273
                          ... 
    Nehru Nagar             11
    Banjara Layout          11
    LB Shastri Nagar        11
    Pattandur Agrahara      11
    Narayanapura            11
    Name: location, Length: 242, dtype: int64
:::
:::

::: {#f97236f1 .cell .code execution_count="525"}
``` python
data.describe()
```

::: {.output .execute_result execution_count="525"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
      <th>price_per_sqft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>13274.000000</td>
      <td>13320.000000</td>
      <td>13320.000000</td>
      <td>13320.000000</td>
      <td>1.327400e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1559.626694</td>
      <td>2.688814</td>
      <td>112.565627</td>
      <td>2.802778</td>
      <td>7.907501e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1238.405258</td>
      <td>1.338754</td>
      <td>148.971674</td>
      <td>1.294496</td>
      <td>1.064296e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>2.678298e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1100.000000</td>
      <td>2.000000</td>
      <td>50.000000</td>
      <td>2.000000</td>
      <td>4.266865e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1276.000000</td>
      <td>2.000000</td>
      <td>72.000000</td>
      <td>3.000000</td>
      <td>5.434306e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1680.000000</td>
      <td>3.000000</td>
      <td>120.000000</td>
      <td>3.000000</td>
      <td>7.311746e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>52272.000000</td>
      <td>40.000000</td>
      <td>3600.000000</td>
      <td>43.000000</td>
      <td>1.200000e+07</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#694239d6 .cell .code execution_count="526"}
``` python
(data['total_sqft']/data['bhk']).describe()
```

::: {.output .execute_result execution_count="526"}
    count    13274.000000
    mean       575.074878
    std        388.205175
    min          0.250000
    25%        473.333333
    50%        552.500000
    75%        625.000000
    max      26136.000000
    dtype: float64
:::
:::

::: {#335baba9 .cell .code execution_count="527"}
``` python
#remove those flats whose (data['total_sqft']/data['bhk']) is less than 300ft
data=data[((data['total_sqft']/data['bhk'])>=300)]
data.describe()
```

::: {.output .execute_result execution_count="527"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
      <th>price_per_sqft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12530.000000</td>
      <td>12530.000000</td>
      <td>12530.000000</td>
      <td>12530.000000</td>
      <td>12530.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1594.564544</td>
      <td>2.559537</td>
      <td>111.382401</td>
      <td>2.650838</td>
      <td>6303.979357</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1261.271296</td>
      <td>1.077938</td>
      <td>152.077329</td>
      <td>0.976678</td>
      <td>4162.237981</td>
    </tr>
    <tr>
      <th>min</th>
      <td>300.000000</td>
      <td>1.000000</td>
      <td>8.440000</td>
      <td>1.000000</td>
      <td>267.829813</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1116.000000</td>
      <td>2.000000</td>
      <td>49.000000</td>
      <td>2.000000</td>
      <td>4210.526316</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1300.000000</td>
      <td>2.000000</td>
      <td>70.000000</td>
      <td>3.000000</td>
      <td>5294.117647</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1700.000000</td>
      <td>3.000000</td>
      <td>115.000000</td>
      <td>3.000000</td>
      <td>6916.666667</td>
    </tr>
    <tr>
      <th>max</th>
      <td>52272.000000</td>
      <td>16.000000</td>
      <td>3600.000000</td>
      <td>16.000000</td>
      <td>176470.588235</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#dce4b9a4 .cell .code execution_count="528"}
``` python
data.shape
```

::: {.output .execute_result execution_count="528"}
    (12530, 7)
:::
:::

::: {#fdc3a114 .cell .code execution_count="529"}
``` python
data.price_per_sqft.describe()
```

::: {.output .execute_result execution_count="529"}
    count     12530.000000
    mean       6303.979357
    std        4162.237981
    min         267.829813
    25%        4210.526316
    50%        5294.117647
    75%        6916.666667
    max      176470.588235
    Name: price_per_sqft, dtype: float64
:::
:::

::: {#a57836fd .cell .code execution_count="530"}
``` python
 #remove flats having large price per sq ft
def remove_outliers_sqft(df):
    df_output= pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        
        gen_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_output=pd.concat([df_output,gen_df],ignore_index=True)
    return df_output
data=remove_outliers_sqft(data)
data.describe()
    
        
```

::: {.output .execute_result execution_count="530"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
      <th>price_per_sqft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10301.000000</td>
      <td>10301.000000</td>
      <td>10301.000000</td>
      <td>10301.000000</td>
      <td>10301.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1508.440608</td>
      <td>2.471702</td>
      <td>91.286372</td>
      <td>2.574896</td>
      <td>5659.062876</td>
    </tr>
    <tr>
      <th>std</th>
      <td>880.694214</td>
      <td>0.979449</td>
      <td>86.342786</td>
      <td>0.897649</td>
      <td>2265.774749</td>
    </tr>
    <tr>
      <th>min</th>
      <td>300.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1250.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1110.000000</td>
      <td>2.000000</td>
      <td>49.000000</td>
      <td>2.000000</td>
      <td>4244.897959</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1286.000000</td>
      <td>2.000000</td>
      <td>67.000000</td>
      <td>2.000000</td>
      <td>5175.600739</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1650.000000</td>
      <td>3.000000</td>
      <td>100.000000</td>
      <td>3.000000</td>
      <td>6428.571429</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30400.000000</td>
      <td>16.000000</td>
      <td>2200.000000</td>
      <td>16.000000</td>
      <td>24509.803922</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#31b526e3 .cell .code execution_count="531"}
``` python
def bhk_outlier_remover(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1) 
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
```
:::

::: {#462d570a .cell .code execution_count="532"}
``` python
data=bhk_outlier_remover(data)
```
:::

::: {#2ea756ee .cell .code execution_count="533"}
``` python
data.shape
```

::: {.output .execute_result execution_count="533"}
    (7361, 7)
:::
:::

::: {#cc6cec23 .cell .code execution_count="534"}
``` python
data
```

::: {.output .execute_result execution_count="534"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>size</th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
      <th>price_per_sqft</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1st Block Jayanagar</td>
      <td>4 BHK</td>
      <td>2850.0</td>
      <td>4.0</td>
      <td>428.0</td>
      <td>4</td>
      <td>15017.543860</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1st Block Jayanagar</td>
      <td>3 BHK</td>
      <td>1630.0</td>
      <td>3.0</td>
      <td>194.0</td>
      <td>3</td>
      <td>11901.840491</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1st Block Jayanagar</td>
      <td>3 BHK</td>
      <td>1875.0</td>
      <td>2.0</td>
      <td>235.0</td>
      <td>3</td>
      <td>12533.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1st Block Jayanagar</td>
      <td>3 BHK</td>
      <td>1200.0</td>
      <td>2.0</td>
      <td>130.0</td>
      <td>3</td>
      <td>10833.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1st Block Jayanagar</td>
      <td>2 BHK</td>
      <td>1235.0</td>
      <td>2.0</td>
      <td>148.0</td>
      <td>2</td>
      <td>11983.805668</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10292</th>
      <td>other</td>
      <td>2 BHK</td>
      <td>1200.0</td>
      <td>2.0</td>
      <td>70.0</td>
      <td>2</td>
      <td>5833.333333</td>
    </tr>
    <tr>
      <th>10293</th>
      <td>other</td>
      <td>1 BHK</td>
      <td>1800.0</td>
      <td>1.0</td>
      <td>200.0</td>
      <td>1</td>
      <td>11111.111111</td>
    </tr>
    <tr>
      <th>10296</th>
      <td>other</td>
      <td>2 BHK</td>
      <td>1353.0</td>
      <td>2.0</td>
      <td>110.0</td>
      <td>2</td>
      <td>8130.081301</td>
    </tr>
    <tr>
      <th>10297</th>
      <td>other</td>
      <td>1 Bedroom</td>
      <td>812.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>1</td>
      <td>3201.970443</td>
    </tr>
    <tr>
      <th>10300</th>
      <td>other</td>
      <td>4 BHK</td>
      <td>3600.0</td>
      <td>5.0</td>
      <td>400.0</td>
      <td>4</td>
      <td>11111.111111</td>
    </tr>
  </tbody>
</table>
<p>7361 rows × 7 columns</p>
</div>
```
:::
:::

::: {#24c9c863 .cell .code execution_count="535"}
``` python
 #price_per_sqft was just remove ouliers so drop it
data.drop(columns=['size','price_per_sqft'],inplace=True)
```
:::

::: {#e6957eda .cell .code execution_count="536"}
``` python
data.head()
```

::: {.output .execute_result execution_count="536"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>total_sqft</th>
      <th>bath</th>
      <th>price</th>
      <th>bhk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1st Block Jayanagar</td>
      <td>2850.0</td>
      <td>4.0</td>
      <td>428.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1st Block Jayanagar</td>
      <td>1630.0</td>
      <td>3.0</td>
      <td>194.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1st Block Jayanagar</td>
      <td>1875.0</td>
      <td>2.0</td>
      <td>235.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1st Block Jayanagar</td>
      <td>1200.0</td>
      <td>2.0</td>
      <td>130.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1st Block Jayanagar</td>
      <td>1235.0</td>
      <td>2.0</td>
      <td>148.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#acbbd753 .cell .code execution_count="537"}
``` python
data.to_csv("Cleaned_data.csv")
```
:::

::: {#bb156601 .cell .code execution_count="538"}
``` python
X=data.drop(columns=['price'])
y=data['price']
```
:::

::: {#d37a16b3 .cell .code execution_count="539"}
``` python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
```
:::

::: {#8b78342b .cell .code execution_count="540"}
``` python
#80% data in X_train and 20%in X_test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
```
:::

::: {#b1f10a6d .cell .markdown}
print(X_train.shape) print(X_test.shape)
:::

::: {#0315cac4 .cell .code execution_count="541"}
``` python
column_trans=make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')
```
:::

::: {#ff1cac20 .cell .code execution_count="542"}
``` python
scaler=StandardScaler()
```
:::

::: {#0baeb413 .cell .code execution_count="543"}
``` python
lr=LinearRegression(normalize=True)
```
:::

::: {#fb5476b6 .cell .code execution_count="544"}
``` python
pipe=make_pipeline(column_trans,scaler,lr)
```
:::

::: {#d40edb60 .cell .code execution_count="545"}
``` python
pipe.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="545"}
    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('onehotencoder',
                                                      OneHotEncoder(sparse=False),
                                                      ['location'])])),
                    ('standardscaler', StandardScaler()),
                    ('linearregression', LinearRegression(normalize=True))])
:::
:::

::: {#9f8441de .cell .code execution_count="546"}
``` python
y_pred_lr=pipe.predict(X_test)
```
:::

::: {#97a6da01 .cell .code execution_count="547"}
``` python
r2_score(y_test,y_pred_lr)
```

::: {.output .execute_result execution_count="547"}
    0.8233777060582834
:::
:::

::: {#2be1ddf7 .cell .code execution_count="548"}
``` python
 lasso=Lasso()
```
:::

::: {#7f1e31bc .cell .code execution_count="549"}
``` python
pipe=make_pipeline(column_trans,scaler,lasso)
```
:::

::: {#affe49d9 .cell .code execution_count="550"}
``` python
pipe.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="550"}
    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('onehotencoder',
                                                      OneHotEncoder(sparse=False),
                                                      ['location'])])),
                    ('standardscaler', StandardScaler()), ('lasso', Lasso())])
:::
:::

::: {#1cc07a2d .cell .code execution_count="551"}
``` python
y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)
```

::: {.output .execute_result execution_count="551"}
    0.8128285650772719
:::
:::

::: {#a6723a0f .cell .code execution_count="552"}
``` python
ridge=Ridge()
```
:::

::: {#9e602e40 .cell .code execution_count="553"}
``` python
pipe=make_pipeline(column_trans,scaler,ridge)
```
:::

::: {#b264573e .cell .code execution_count="554"}
``` python
pipe.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="554"}
    Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder='passthrough',
                                       transformers=[('onehotencoder',
                                                      OneHotEncoder(sparse=False),
                                                      ['location'])])),
                    ('standardscaler', StandardScaler()), ('ridge', Ridge())])
:::
:::

::: {#e3681977 .cell .code execution_count="555"}
``` python
y_pred_ridge=pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)
```

::: {.output .execute_result execution_count="555"}
    0.82341466333127
:::
:::

::: {#5b27591e .cell .code execution_count="557"}
``` python
print("No Regularization",r2_score(y_test,y_pred_lr))
print("Lasso",r2_score(y_test,y_pred_lasso))
print("Ridge",r2_score(y_test,y_pred_ridge))
```

::: {.output .stream .stdout}
    No Regularization 0.8233777060582834
    Lasso 0.8128285650772719
    Ridge 0.82341466333127
:::
:::

::: {#fae6f4fc .cell .code execution_count="558"}
``` python
import pickle
```
:::

::: {#a34c9df5 .cell .code execution_count="559"}
``` python
pickle.dump(pipe,open('RidgeModel.pkl','wb'))
```
:::

::: {#5528547a .cell .code}
``` python
```
:::
