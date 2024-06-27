FROM incebellipipo/devcontainer:jammy

# Copy python package dependencies
COPY requirements.txt /tmp/requirements.txt
COPY requirements.examples.txt /tmp/requirements.examples.txt

# Install python package dependencies
RUN pip install -r /tmp/requirements.txt
RUN pip install -r /tmp/requirements.examples.txt

# Set environment variables to seamlessly work with python
ENV PYTHONPATH="${PYTHONPATH}:/com.docker.devenvironments.code/src"
