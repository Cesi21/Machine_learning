import csv
import os

input_csv = 'C:/Users/yourname/Desktop/tretji zagovor/kaggle_sibmission.csv'

with open(input_csv, mode='r', newline='', encoding='utf-8') as infile, \
        open(input_csv, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        filename = os.path.basename(row[0])
        writer.writerow([filename, row[1]])