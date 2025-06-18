# Use Miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Copy Conda environment file
COPY environment.yml .

# Update Conda separately to avoid conflicts
RUN conda update -n base -c defaults conda -y

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment
RUN echo "conda activate yolo" >> ~/.bashrc

# Set default command
CMD ["bash"]
