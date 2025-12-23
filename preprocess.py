import numpy
import pandas as pd

# identify the statistic of diabetes data
def statistics(df):
    left_none, left_risky, left_high = 0, 0, 0
    right_none, right_risky, right_high = 0, 0, 0
    for index, row in df.iterrows():
        if 'nan' not in str(row['图片1']):
            #left_eye.append(index)
            if row['图片1的ETDRS1'] > 50:
                left_high += 1
            elif row['图片1的ETDRS1'] < 20:
                left_none += 1
            else:
                left_risky += 1
        if 'nan' not in str(row['图片2']):
            #right_eye.append(index)
            if row['图片2的ETDRS2'] > 50:
                right_high += 1
            elif row['图片2的ETDRS2'] < 20:
                right_none += 1
            else:
                right_risky += 1
    valids = left_none + left_risky + left_high + right_none + right_risky + right_high 
    benigns = left_none + right_none
    print(f" validate overall images : {valids} benigns {benigns} left benigns {left_none} right benigns {right_none}")
    overall = left_risky + left_high + right_risky + right_high
    print(f" the overall of diabetes : {overall}")
    lefts = left_risky + left_high
    print(f" left eye diabetes : {lefts} very high risk {left_high}")
    rights = right_risky + right_high
    print(f" right eye diabetes : {rights} very high risk {right_high}")


if __name__ == "__main__":
    df = pd.read_excel('file.xlsx')
    left_eye, right_eye = [], []
    for index, row in df.iterrows():
        if 'nan' not in str(row['图片1']):
            left_eye.append(index)
        if 'nan' not in str(row['图片2']):
            right_eye.append(index)

    