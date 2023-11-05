from matplotlib import pyplot as plt

def plot_history(history, smoothing = False): 
    
    if not smoothing:
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
    else:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc)+1)


        
        plt.plot(epochs, smooth_curve(acc), 'bo', label = 'Training acc')
        plt.plot(epochs, smooth_curve(val_acc), 'b', label = 'Validation Accuracy')
        plt.title('SMOOTHED Training and validation accruacy')
        plt.legend()
        plt.savefig('SMOOTHED Training and valdation accuracy.png')
        plt.figure()

        plt.plot(epochs, smooth_curve(loss), 'bo', label = 'Training loss')
        plt.plot(epochs, smooth_curve(val_loss), 'b', label = 'Validation loss')
        plt.title('SMOOTHED Training and valdation loss')
        plt.legend()

        plt.savefig('SMOOTHED Training and valdation loss.png')



def smooth_curve(points, factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else: 
            smoothed_points.append(point)
    return smoothed_points
    
