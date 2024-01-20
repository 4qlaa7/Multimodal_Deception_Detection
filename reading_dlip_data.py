import csv
import numpy as np

TrainF = []
TrainL = []
TestF = []
TestL = []

# Assign Path for deceptive
Facial_Features = '/content/drive/MyDrive/Graduation Project/Features/Facial_Deceptive_features_dlib.csv'
Audio_Features = '/content/drive/MyDrive/Graduation Project/Features/Deceptive_Audio_Features.csv'
Trans_Features = '/content/drive/MyDrive/Graduation Project/Features/Deceptive_Transcript_Features.csv'

# Assign path for truthful
Facial_Features2 = '/content/drive/MyDrive/Graduation Project/Features/Facial_Truthful_features_dlib.csv'
Audio_Features2 = '/content/drive/MyDrive/Graduation Project/Features/Truthful_Audio_Features.csv'
Trans_Features2 = '/content/drive/MyDrive/Graduation Project/Features/Truthful_Transcript_Features.csv'


# Read Deceptive
for i in range(40):
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
    facialDeception.append(float(row[t]))
    t += 1
  facialDeception = np.asarray(facialDeception).astype(np.float32)


  with open(Audio_Features, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break

  t = 1
  AudioDeception = []
  while t < len(row):
    if float(row[t]) > 0 or float(row[t]) < 0:
      AudioDeception.append(float(row[t]))
    t += 1
  AudioDeception = np.asarray(AudioDeception).astype(np.float32)

  with open(Trans_Features, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break
  t = 1
  TransDeception = []
  while t < len(row):
    if row[t]:
      TransDeception.append(float(row[t]))
    t += 1
  TransDeception = np.asarray(TransDeception).astype(np.float32)
  arr = np.concatenate((facialDeception,AudioDeception,TransDeception))
  arr = np.array(arr)
  TrainF.append(arr)

print(len(TrainF))

for j in range(len(TrainF)):
  TrainL.append(1)


print(len(TrainL))


for i in range(40,51):
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
    facialDeception.append(float(row[t]))
    t += 1
  facialDeception = np.asarray(facialDeception).astype(np.float32)


  with open(Audio_Features, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break

  t = 1
  AudioDeception = []
  while t < len(row):
    if float(row[t]) > 0 or float(row[t]) < 0:
      AudioDeception.append(float(row[t]))
    t += 1
  AudioDeception = np.asarray(AudioDeception).astype(np.float32)

  with open(Trans_Features, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break
  t = 1
  TransDeception = []
  while t < len(row):
    if row[t]:
      TransDeception.append(float(row[t]))
    t += 1
  TransDeception = np.asarray(TransDeception).astype(np.float32)
  arr = np.concatenate((facialDeception,AudioDeception,TransDeception))
  arr = np.array(arr)
  TestF.append(arr)

print("Test")
print(len(TestF))


for j in range(len(TestF)):
  TestL.append(1)

print(len(TestL))


# Read Truthful



for i in range(40):
  arr = []
  desired_row_number = i + 1
  with open(Facial_Features2, 'r') as csv_file:
      csv_reader = csv.reader(csv_file)
      for current_row_number, row in enumerate(csv_reader, start=0):
          if current_row_number == desired_row_number:
              break
  t = 1
  facialDeception = []
  while t < len(row):
    facialDeception.append(float(row[t]))
    t += 1
  facialDeception = np.asarray(facialDeception).astype(np.float32)


  with open(Audio_Features2, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break

  t = 1
  AudioDeception = []
  while t < len(row):
    if float(row[t]) > 0 or float(row[t]) < 0:
      AudioDeception.append(float(row[t]))
    t += 1
  AudioDeception = np.asarray(AudioDeception).astype(np.float32)

  with open(Trans_Features2, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break
  t = 1
  TransDeception = []
  while t < len(row):
    if row[t]:
      TransDeception.append(float(row[t]))
    t += 1
  TransDeception = np.asarray(TransDeception).astype(np.float32)
  arr = np.concatenate((facialDeception,AudioDeception,TransDeception))

  arr = np.array(arr)
  TrainF.append(arr)


print(len(TrainF))

for j in range(40,len(TrainF)):
  TrainL.append(0)

print(len(TrainL))

for i in range(40,51):
  arr = []
  desired_row_number = i + 1
  with open(Facial_Features2, 'r') as csv_file:
      csv_reader = csv.reader(csv_file)
      for current_row_number, row in enumerate(csv_reader, start=0):
          if current_row_number == desired_row_number:
              break
  t = 1
  facialDeception = []
  while t < len(row):
    facialDeception.append(float(row[t]))
    t += 1
  facialDeception = np.asarray(facialDeception).astype(np.float32)


  with open(Audio_Features2, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break

  t = 1
  AudioDeception = []
  while t < len(row):
    if float(row[t]) > 0 or float(row[t]) < 0:
      AudioDeception.append(float(row[t]))
    t += 1
  AudioDeception = np.asarray(AudioDeception).astype(np.float32)

  with open(Trans_Features2, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for current_row_number, row in enumerate(csv_reader, start=0):
        if current_row_number == desired_row_number:
            break
  t = 1
  TransDeception = []
  while t < len(row):
    if row[t]:
      TransDeception.append(float(row[t]))
    t += 1
  TransDeception = np.asarray(TransDeception).astype(np.float32)
  arr = np.concatenate((facialDeception,AudioDeception,TransDeception))

  arr = np.array(arr)
  TestF.append(arr)


print("Test")
print(len(TestF))


for j in range(11,len(TestF)):
  TestL.append(0)

print(len(TestL))

# Check if there any missing values ot tuples 
missing_values = any(any(v is None or isinstance(v, tuple) for v in sublist) for sublist in TestF)

if missing_values:
  print("There are Tuples and Missing Values")
else:
  print("No Missing Values")



# Padd the arrays
  
def paddd(arrays):
    max_length = 636228 #Because it's the maximum number for all the videos
    padded_arrays = [np.append(array,[0] * (max_length - len(array)),axis=None) for array in arrays]
    print(max_length)
    return padded_arrays


New_Test = paddd(TestF)
New_Train = paddd(TrainF)


# Converting To ndArrays

New_Train = np.asarray(New_Train).astype(np.float32)
New_Test = np.asarray(New_Test).astype(np.float32)
TrainL = np.asarray(TrainL).astype(np.float32)
TestL = np.asarray(TestL).astype(np.float32)


print("TrainF shape:", New_Train.shape)
print("TrainL shape:", TrainL.shape)
print("TestF shape:", New_Test.shape)
print("TestL shape:", TestL.shape)
print("TrainF type:", New_Train.dtype)
print("TrainL type:", TrainL.dtype)
print("TestF type:", New_Test.dtype)
print("TestL type:", TestL.dtype)

# Saving into numpy files

np.savez('Padded_Training.npz', features=New_Train, labels=TrainL)
np.savez('Padded_Testing.npz', features=New_Test, labels=TestL)