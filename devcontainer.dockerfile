FROM incebellipipo/devcontainer:jammy

# Copy python package dependencies
COPY requirements.txt /tmp/requirements.txt

# Install python package dependencies
RUN pip install -r /tmp/requirements.txt

# Set environment variables to seamlessly work with python
ENV PYTHONPATH="${PYTHONPATH}:/com.docker.devenvironments.code/src"