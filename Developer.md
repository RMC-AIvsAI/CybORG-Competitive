*Under Construction*

## Install the Competitive-CybORG environment

**Important to start from a fresh conda environment. Don’t install `requirements.txt` before installing and confirming Tensorflow+GPU working.**

1. Follow the guide below to install miniconda, tensorflow, etc. for your OS and environment.
   - https://www.tensorflow.org/install/pip#step-by-step_instructions
   - e.g. For Windows 10, I needed tensorflow < 2.11

2. Confirm Tensorflow + GPU is working:

    ```
    (competitive-cyborg-env) ~> python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    
    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    ```

3. Install other Competitive-CybORG libraries using `pip install -r CybORG/Requirements.txt`

4. To learn about the original CybORG environment, you can explore the original repository, install it and follow the tutorial
   - https://github.com/cage-challenge/CybORG/tree/main/CybORG/Tutorial
   - _Make sure you create/activate a new conda environment so you don't interfere with the Competitive-CybORG setup_

