import function_folder.librarys as lib

def file_upload():

    """import_data_and_viusalization .ipynb

    Automatically generated by Colaboratory.

    Original file is located at
        https://colab.research.google.com/drive/1qnAtVJOQ93PBrTubts7IsPhmQQ_0dwDd

        npzobj is an object of the class npzfile. by accessing [data] we have a tensor of this sizes:
    """

 
    # Commented out IPython magic to ensure Python compatibility.
    # %pwd
    # %cd drive/MyDrive

    npzobj = lib.np.load('public_data.npz', allow_pickle=True)

    npzobj.files

    print(npzobj['data'].shape)
    print(npzobj['labels'].shape)

    immagine = npzobj['data'][45]
    immagine_normalizzata = immagine.astype(lib.np.float32) / 255.0  # Normalizzazione dei valori tra 0 e 1

    # Plot dell'immagine RGB
    lib.plt.imshow(immagine_normalizzata)
    string = str ( npzobj['labels'][45])
    lib.plt.title(string)
    lib.plt.axis('off')  # Rimuove gli assi
    lib.plt.show()
    return (npzobj)
