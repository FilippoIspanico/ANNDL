from matplotlib import pyplot as plt

def plot_history(history): 
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
    plt.title('Training and validation accruacy')
    plt.legend()
    plt.savefig('Training and valdation accuracy.png')
    plt.figure()

    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and valdation loss')
    plt.legend()

    plt.savefig('Training and valdation loss.png')
