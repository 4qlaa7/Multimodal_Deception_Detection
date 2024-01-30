import csv
import numpy as np

def Read_Data(i,Facial_Features):
    arr = []
    desired_row_number = i + 1
    with open(Facial_Features, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for current_row_number, row in enumerate(csv_reader, start=0):
            if current_row_number == desired_row_number:
                break
    t = 1
    facialDeception = []
    while t < len(row):
      if row[t]:
        facialDeception.append(float(row[t]))
      t += 1
    Data = np.asarray(facialDeception).astype(np.float32)
    return Data


def Padding(arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = [np.append(array,[0] * (max_length - len(array)),axis=None) for array in arrays]
    print(max_length)
    return padded_arrays