import numpy as np
from spec_net.architectures import ann2

mask = np.linspace(0, 3599, 670).astype(int) # to scale down the features

X = np.load("C:/Users/itofischer/Desktop/SpectralAnalysis/data/npy/raman/categorized/mitochondria_data.npy")[:, mask]
y = np.load("C:/Users/itofischer/Desktop/SpectralAnalysis/data/npy/raman/categorized/mitochondria_labels.npy")[:, 1]

attributes = np.linspace(3000, 400, 670)

# optimizer = ann2.FSANN(X, y, 
#                        layer_name = 'Linear', 
#                        n_features = 3,
#                        callback = {'metric': 'accuracy',
#                                    'thresh': 0.99,
#                                    'd_min': 30,
#                                    'd_max': 300,
#                                    'n_samples': 10,
#                                    'decay': 0.0005},
#                        features = attributes, 
#                        architecture_type = 'down',
#                        # normalization = 'standardize', 
#                        activation_function = 'relu')

# optimizer.data.X_train

# optimizer.set_n_layers(4)
# optimizer.set_learning_rate(0.0001)
# optimizer.set_batch_size(16)
# optimizer.set_epochs(2000)

# optimizer.train_model()
# optimizer.plot.history()
# optimizer.model.summary()
# optimizer.plot.mask_history()
# # optimizer.plot.input_reduction()



from spec_net.preprocess.data import synchronous, asynchronous
from spec_net.preprocess import dataformat as df
synchr = synchronous(X, False)
asynchr = asynchronous(X, False)

import matplotlib.pyplot as plt
import numpy as np

X = df.min_max_scale(X)

def correlation_spectrum_2d(arr, x, rowvar=True, mode='synchronous'):

    from spec_net.preprocess.data import synchronous, asynchronous
    
    if not rowvar:
        arr = arr.T
    
    # 2d correlation spectrum
    fig = plt.figure("2D Correlation Spectrum", figsize=(5,5))
    plt.clf()
    gs = fig.add_gridspec(2, 2, 
                          width_ratios=[1,5], wspace=0.1,
                          height_ratios=[1,5], hspace=0.1)

    ax2 = fig.add_subplot(gs[0,1], label='HorizontalSpectrum')
    ax3 = fig.add_subplot(gs[1,0], label='VerticalSpectrum')
    ax4 = fig.add_subplot(gs[1,1], label='2DCorrelation')
    
    mu = np.mean(arr, axis = 1, keepdims = True)

    ax2.plot(x, mu)
    ax2.set_xlim(x[0], x[-1])
    ax2.grid(True)
    ax3.plot(mu, x)
    ax3.set_ylim(x[0], x[-1])
    ax3.grid(True)
    
    ax2.set_yticks([])
    ax2.xaxis.set_ticks_position('top')
    ax3.set_xticks([])
    
    ax3.invert_yaxis()
    ax3.invert_xaxis()
    
    if mode == 'synchronous':
        corr = synchronous(arr)
    elif mode == 'asynchronous':
        corr = asynchronous(arr)
    else:
        raise ValueError(f"There is no mode called '{mode}'. Try 'synchronous'"
                         "or 'asynchronous' instead.")
    
    ax4.imshow(corr, extent=[x[0], x[-1], x[-1], x[0]], aspect=1, cmap='Blues')
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.tight_layout()
    
correlation_spectrum_2d(X, 
                        attributes, 
                        rowvar=False, 
                        mode='synchronous')
