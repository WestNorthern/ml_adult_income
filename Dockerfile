FROM jupyter/datascience-notebook:latest

# 1) Become root so we can install & set perms
USER root

# 2) Copy and install your deps; then clean up that file
COPY environment.yml /tmp/
RUN mamba env update -n base -f /tmp/environment.yml \
 && rm /tmp/environment.yml \
 && fix-permissions /home/jovyan

# 3) Switch back to the jovyan user
USER jovyan

# 4) Bring in your project and set the workdir
COPY . /home/jovyan/project
WORKDIR /home/jovyan/project

EXPOSE 8888

# 5) Start JupyterLab
CMD ["start.sh", "jupyter", "lab", "--LabApp.token=''", "--LabApp.allow_origin='*'"]
