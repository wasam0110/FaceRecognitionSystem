import sys
import types
import tensorflow as tf

# --- Create a simulated tensorflow.keras package with all expected submodules ---
if "tensorflow.keras" not in sys.modules:
    keras_pkg = types.ModuleType("tensorflow.keras")
    keras_pkg.__path__ = []  # mark as package
    keras_pkg.__dict__.update(tf.keras.__dict__)
    sys.modules["tensorflow.keras"] = keras_pkg

    # Register commonly used submodules
    submodules = [
        "models",
        "layers",
        "optimizers",
        "losses",
        "backend",
        "callbacks",
        "applications",
        "utils"
    ]

    for sub in submodules:
        mod = types.ModuleType(f"tensorflow.keras.{sub}")
        try:
            tf_sub = getattr(tf.keras, sub)
            mod.__dict__.update(tf_sub.__dict__)
        except Exception:
            pass
        sys.modules[f"tensorflow.keras.{sub}"] = mod

# --- Add the missing 'tensorflow.keras.preprocessing' shim ---
if "tensorflow.keras.preprocessing" not in sys.modules:
    preprocessing_pkg = types.ModuleType("tensorflow.keras.preprocessing")
    preprocessing_pkg.__path__ = []

    # TensorFlow 2.15 moved image utilities under tf.keras.utils
    # Expose them for backward compatibility
    image_mod = types.ModuleType("tensorflow.keras.preprocessing.image")
    try:
        image_mod.img_to_array = tf.keras.utils.img_to_array
        image_mod.array_to_img = tf.keras.utils.array_to_img
        image_mod.load_img = tf.keras.utils.load_img
    except Exception:
        pass

    preprocessing_pkg.image = image_mod
    sys.modules["tensorflow.keras.preprocessing"] = preprocessing_pkg
    sys.modules["tensorflow.keras.preprocessing.image"] = image_mod

print("✅ tensorflow.keras package fully restored (including .preprocessing.image)")
