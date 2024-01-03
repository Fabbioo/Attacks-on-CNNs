from .imports import warnings, torch, plt

warnings.filterwarnings('ignore')

device: str = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

x_img_resize: int = 224
y_img_resize: int = 224
img_resize: tuple = (x_img_resize, y_img_resize)

x_figure_size: int = 16
y_figure_size: int = 8
fig_size: tuple = (x_figure_size, y_figure_size)

parametri_grafici = {
    'figure.figsize': fig_size, # Dimensione della figura.
    'figure.autolayout': True,  # Regolazione automatica delle dimensioni della figura.
    'figure.titlesize': 20,     # Dimensione del titolo associato ad ogni figura (plt.suptitle()).
    'axes.titlesize': 20,       # Dimensione del titolo associato ad ogni grafico all'interno di una figura (plt.title()).
    'axes.labelsize': 20,       # Dimensione delle etichette sia sull'asse x sia sull'asse y.
    'xtick.labelsize': 15,      # Dimensione dei riferimenti sull'asse x.
    'ytick.labelsize': 15,      # Dimensione dei riferimenti sull'asse y.
    'legend.fontsize': 20,      # Dimensione dei caratteri della legenda.
}
plt.rcParams.update(parametri_grafici)