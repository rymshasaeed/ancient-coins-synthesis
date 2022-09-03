import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define a function to prepare the dataset
def dataset(path):
    # Load the dataset info
    df = pd.read_csv('../data/query.csv')[['RecordId', 'Denomination']]

    # Create an empty dataframe to store the dataset
    data = pd.DataFrame(columns=['images', 'denomination'])

    # Get all subfolders
    for folder in os.listdir(path):
        # Find denomination
        idx = np.where(df['RecordId'] == folder)[0][0]
        d = df['Denomination'][idx]

        # Get all images files
        for filename in os.listdir(os.path.join(path, folder)):
            # Get all image paths
            img = os.path.join(path, folder, filename)

            # Store results in a dataframe
            data = data.append({'images': img, 'denomination': d}, ignore_index=True)

    # Encode denominations as numeric labels
    data['labels'] = data['denomination'].astype('category').cat.codes

    return data

# Save dataset to local directory
df = dataset('../images/')
df.to_csv('../data/coins-dataset.csv', index=False)

# Visualize dataset distribution
p = sns.countplot(x="denomination", data=df, palette='Paired')
p.set_xticklabels(p.get_xticklabels(), rotation=90)
plt.title('Dataset distribution')
plt.show()

# Split dataset into train (75%) and test (25%) sets
train = df.sample(frac=0.75, random_state=394)
test = df.drop(train.index)

# Resmaple the training set to balance the coin distribution
# This would avoid the class-imbalance problem during training
class_size = max(train['denomination'].value_counts())
train = train.groupby(['denomination']).apply(lambda x: x.sample(class_size, replace=True)).reset_index(drop=True)
train = train.sample(frac=1).reset_index(drop=True)

# Save split-dataframes to local directory
train.to_csv('../data/Train.csv', index=False)
test.to_csv('../data/Test.csv', index=False)

# Visualize training set distribution
p = sns.countplot(x="denomination", data=train, palette='Paired')
p.set_xticklabels(p.get_xticklabels(), rotation=90)
plt.title('Training set distribution')
plt.show()