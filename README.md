# ANDURYL

-ANDURYL_main.py: programa principal desde el cual se va llamando al resto.
-hamiltonian_class.py: es donde se construirá la parte del programa que emplea el método bayesiano, aunque aún está en proceso.
-priors.py: es donde están definidos los priors, aunque actualmente solo hay uno (el uniforme).
-profiles_torch.py: es donde están definidos los distintos modelos de galaxias, Sersic, exponencial, etc.
-profiles_select.py: llama al programa anterior y construye la imagen del modelo, seguramente en un futuro estos dos últimos se unan. 
-psf_generation.py: genera la psf que se le indique.
-read_params.py: lee los parámetros  del fichero de parametros que es setup.txt.
