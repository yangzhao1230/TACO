# Install PyTorch and related packages
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Clone and install Flash Attention
git clone https://github.com/Dao-AILab/flash-attention.git
pip install flash-attn --no-build-isolation

# Install layer_norm from Flash Attention
cd flash-attention/csrc/layer_norm
pip install .
cd ../..
cd ..

# Install additional dependencies
pip install pytorch_lightning==1.8.6 --no-deps
pip install PyTDC --no-deps
pip install pynvml
pip install lightning_utilities
pip install torchmetrics
pip install tensorboardX
pip install enformer_pytorch
pip install pymemesuite
pip install opt_einsum
pip install pandas 
pip install fuzzywuzzy
pip install scikit-learn
pip install wandb