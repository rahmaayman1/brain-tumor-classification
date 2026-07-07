import os 

#model paths
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    'models/transfer_best_phase2.keras'
)

#Image settings
IMAGE_SIZE = (224,224)
CHANNELS = 3

#Classes
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    'glioma': 'Tumor originating from glial cells within brain tissue',
    'meningioma': 'Tumor originating from the meninges surrounding the brain',
    'notumor': 'No tumor detected',
    'pituitary': 'Tumor in the pituitary gland at the base of the brain'
}

