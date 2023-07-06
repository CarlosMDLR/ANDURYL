

![logoanduryl](https://github.com/CarlosMDLR/ANDURYL/assets/105994653/2d21068f-395e-4d4b-9c76-bed0bcc80624)

ANDURYL (A bayesiaN Decomposition code for Use in photometRY Labors) is a code created to perform photometric decompositions of galaxies using Bayesian statistics.
## Instalation
To correctly install ANDURYL you simply have to clone this repository, and then run 
the requirements.txt file, which includes all the Python packages necessary for the correct operation of the code:

```
git clone https://github.com/CarlosMDLR/ANDURYL.git
python -m pip install -r requirements.txt
```

-ANDURYL_main.py: programa principal desde el cual se va llamando al resto.

-hamiltonian_class.py: es donde se construirá la parte del programa que emplea el método bayesiano, aunque aún está en proceso.

-priors.py: es donde están definidos los priors, aunque actualmente solo hay uno (el uniforme).

-profiles_torch.py: es donde están definidos los distintos modelos de galaxias, Sersic, exponencial, etc.

-profiles_select.py: llama al programa anterior y construye la imagen del modelo, seguramente en un futuro estos dos últimos se unan. 

-psf_generation.py: genera la psf que se le indique.

-read_params.py: lee los parámetros  del fichero de parametros que es setup.txt.
