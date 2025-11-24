# Dockerfile pour Analyse XAS Mayotte
# Base: Python 3.10 avec Ubuntu 22.04

FROM python:3.10-slim

# Métadonnées
LABEL maintainer="lelosq@ipgp.fr"
LABEL description="Container Docker pour analyse XAS des échantillons volcaniques de Mayotte"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

    # Installer les dépendances système nécessaires
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        gfortran \
        libhdf5-dev \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 -s /bin/bash xasuser && \
    mkdir -p /home/xasuser/mayotte && \
    chown -R xasuser:xasuser /home/xasuser

# Définir le répertoire de travail
WORKDIR /home/xasuser/mayotte

# Copier le fichier requirements.txt en premier (pour cache Docker)
COPY --chown=xasuser:xasuser requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copier tout le code source
COPY --chown=xasuser:xasuser . .

# Compiler le code Fortran ctsfg6
RUN cd src && \
    gfortran ctsfg6.for -o ctsfg6 && \
    chmod +x ctsfg6 && \
    cd ..

# Créer les dossiers de sortie
RUN mkdir -p figures/Iron figures/Sulfur figures/Modelling results/modelling && \
    chown -R xasuser:xasuser figures results

# Passer à l'utilisateur non-root
USER xasuser

# Définir le répertoire de travail sur src/ pour exécution des scripts
WORKDIR /home/xasuser/mayotte/src

# Exposer un port si nécessaire (pour futures extensions web)
# EXPOSE 8080

# Point d'entrée par défaut (permet d'exécuter n'importe quelle commande)
ENTRYPOINT []

# Commande par défaut : analyse XAS
CMD ["python", "analysis_publication.py"]
