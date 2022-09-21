import numpy as np
import pandas as pd

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

print(df.head(3))
print(df.tail(3))

df.to_numpy()

print(df.describe())

print(df.T) #tranpose

print(df.sort_values(by="B")) #sort

df["A"] #select a column
df[0:3] #select 0 to 3 row
df["20130102":"20130104"] #select by index from 20130102 to 20130104

df.loc[:, ["A", "B"]] #select all row two columns by name

df.iloc[3:5, 0:2] #select by position


df > 0 #boolean matrix by condition
df[df > 0]

df2 = df.copy() #copy
df2["E"] = ["one", "one", "two", "three", "four", "three"] #add new column

df2[df2["E"].isin(["two", "four"])] #all column of two rows by value


s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6)) #new df
df2["F"] = s1 #full join by row name
print(df2)


# merge
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})

pd.merge(left, right, on="key") # all cross (1,4) (1,5) (2,4) (2,5)


# OJO: antes verificar que las llaves son unicas para no generar cruces adicionales
left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})

pd.merge(left, right, on="key") # (1,4) (2,5)


# group by
df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

df.groupby("A").sum() # delete B

dfg = df.groupby(["A", "B"]).sum() # in group by two rows, index are tuple now

print(dfg)
print(dfg.columns)
print(dfg.index)


print(dfg.loc['bar', 'one']) #select element in MultiIndex


# pivot
print(df)

table = pd.pivot_table(df, values='C', index=['B'],
                    columns=['A'], aggfunc=np.sum)


print(table)

# multiple agregation
table2 = pd.pivot_table(df, values=['C', 'D'], index=['B'], columns=['A'],
                       aggfunc={'C': np.mean, 'D': [min, max, np.mean]} )

print(table2)

#unpivot
print(table)
print(table.stack())

# time series

rng = pd.date_range("1/1/2012", periods=5, freq="M")
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period() #delete day


#categorical

df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)

df["grade"] = df["raw_grade"].astype("category")

