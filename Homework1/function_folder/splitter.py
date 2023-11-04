def splitting_function(y):
    # Convert labels to one-hot encoding format
    y = tfk.utils.to_categorical(y,2)

    # Split data into train_val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=seed, test_size=60, stratify=np.argmax(y,axis=1))

    # Further split train_val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=seed, test_size=60, stratify=np.argmax(y_train_val,axis=1))

    # Print shapes of the datasets
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return [X_train, X_val,X_test, y_train, y_val,y_test]
