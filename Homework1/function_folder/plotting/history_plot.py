from function_folder.librarys import *
def plot_history(history):
    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(history['val_loss'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15,5))
    plt.plot(history['accuracy'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(history['val_accuracy'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()