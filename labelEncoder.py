from sklearn.preprocessing import LabelEncoder

def encode_labels(labels):
    le = LabelEncoder()
    le.fit(labels)
    encoded_labels = le.transform(labels)
    return encoded_labels, le
