# Use the Jupyter “datascience” stack (includes conda, Python, common DS libs)
FROM jupyter/datascience-notebook:latest

# 1) Become root so we can install & set perms
USER root

# 2) Copy and install your deps; then clean up that file
COPY environment.yml /tmp/
RUN mamba env update -n base -f /tmp/environment.yml \
 && rm /tmp/environment.yml \
 && fix-permissions /home/jovyan

# 3) Copy your project into the container
#    (COPY always runs as root to preserve permissions)
COPY . /home/jovyan/project

# 4) Ensure jovyan owns the project files so pip can write egg-info
RUN chown -R jovyan:users /home/jovyan/project

# 5) Install the project in editable mode (creates egg-info under project)
RUN pip install --no-cache-dir -e /home/jovyan/project

# 6) Switch to non-root user for running JupyterLab
USER jovyan

# 7) Set working directory
WORKDIR /home/jovyan/project

# 8) Make src importable without pip by adding to PYTHONPATH
ENV PYTHONPATH="/home/jovyan/project/src:${PYTHONPATH}"

# 9) Expose JupyterLab port
EXPOSE 8888

# 10) Start JupyterLab on container launch
CMD ["start.sh", "jupyter", "lab", "--LabApp.token=''", "--LabApp.allow_origin='*'"]
