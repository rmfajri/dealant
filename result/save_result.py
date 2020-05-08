import csv
import pandas as pd
text=[]
label=[]
text1=[]
label1=[]
with open('annotator_1/random_forest/labeled_sentences.csv',encoding='utf-8',errors='ignore') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]} {row[1]} .')
            line_count += 1
            label.append(row[0])
            text.append(row[1])
    #print(f'Processed {line_count} lines.')

with open('annotator_1/random_forest/RF1.csv',encoding='utf-8',errors='ignore') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]} {row[1]} .')
            line_count += 1
            label.append(row[0])
            text.append(row[1])


d={'label':label,'text':text}
df=pd.DataFrame(data=d)

print(len(df))
print(df.head())

df.to_csv('annotator_1/random_forest/result.csv',index=False)