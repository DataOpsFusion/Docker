FROM jupyter/datascience-notebook

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    statsmodels \
    plotly \
    dash \
    bokeh \
    altair \
    'tensorflow[and-cuda]' \
    torch \
    torchvision \
    torchaudio \



RUN sudo apt-get install -y \
    r-base \
    r-cran-boot \
    r-cran-class \
    r-cran-cluster \
    r-cran-codetools \
    r-cran-foreign \
    r-cran-kernsmooth \ 
    r-cran-lattice \ 
    r-cran-mass \
    r-cran-matrix \
    r-cran-mgcv \
    r-cran-nlme \
    r-cran-nnet \
    r-cran-rpart \
    r-cran-spatial \
    r-cran-survival \

CMD ["jupyter notebook"]