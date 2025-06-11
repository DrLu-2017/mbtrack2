FROM gitlab-registry.synchrotron-soleil.fr/pa/collective-effects/python_mpi:latest
LABEL name mbtrack2
USER dockeruser
WORKDIR '/home/dockeruser'
COPY --chown=dockeruser mbtrack2 /home/dockeruser/mbtrack2
ENV PYTHONPATH=/home/dockeruser/