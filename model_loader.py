from keras.models import load_model

model_path = 'G:/DHARSHNI_WORKS/Drowsy_driver_detector/models/cnnCat2.keras'

try:
    model = load_model(model_path, custom_objects={})  # Add custom objects if necessary
    print("Model loaded successfully!")
except KeyError as e:
    print(f"KeyError: {e}. Please check the model file.")
except Exception as e:
    print(f"Error loading model: {e}")
