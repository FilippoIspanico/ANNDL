import function_folder.librarys as lib

def splitting_function(X,y):

    def map_labels(y):
        class_mapping = {"healthy": 0, "unhealthy": 1}
        return lib.np.array([class_mapping[label] for label in y])

    # ...
    # Nel tuo codice:

    # Dopo aver caricato y dai dati, mappa le etichette
    y = map_labels(y)
    # Convert labels to one-hot encoding format
    y = lib.tfk.utils.to_categorical(y,2)

    # Split data into train_val and test sets
    X_train_val, X_test, y_train_val, y_test = lib.train_test_split(X, y, random_state=lib.seed, test_size=60, stratify=lib.np.argmax(y,axis=1))

    # Further split train_val into train and validation sets
    X_train, X_val, y_train, y_val = lib.train_test_split(X_train_val, y_train_val, random_state=lib.seed, test_size=60, stratify=lib.np.argmax(y_train_val,axis=1))

    # Print shapes of the datasets
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return [X_train, X_val,X_test, y_train, y_val,y_test]
