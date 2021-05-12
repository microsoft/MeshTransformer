# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/models ] ; then
    mkdir -p $REPO_DIR/models
fi
BLOB='https://datarelease.blob.core.windows.net/metro'

# --------------------------------
# Download our model predictions that can be submitted to FreiHAND Leaderboard
# --------------------------------
if [ ! -d $REPO_DIR/predictions ] ; then
    mkdir -p $REPO_DIR/predictions
fi
# (1) Our model + test-time augmentation. It achieves 6.3 PA-MPVPE on FreiHAND Leaderboard
wget -nc $BLOB/metro_release_ckpt200-multisc-pred.zip -O $REPO_DIR/predictions/ckpt200-multisc-pred.zip
# (2) Our model prediction. It achieves 6.7 PA-MPVPE on FreiHAND Leaderboard
wget -nc $BLOB/metro_release_ckpt200-sc10_rot0-pred.zip -O $REPO_DIR/predictions/ckpt200-sc10_rot0-pred.zip

