import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import numpy as np
from PIL import Image


# Check if the dataset exists and delete it
if "my_custom_vit_dataset" in fo.list_datasets():
    fo.delete_dataset("my_custom_vit_dataset")
    print("Deleted dataset: my_custom_vit_dataset")
else:
    print("Dataset 'my_custom_vit_dataset' does not exist.")

# Reload the dataset (if it wasn't deleted)
if "my_custom_vit_dataset" in fo.list_datasets():
    dataset = fo.load_dataset("my_custom_vit_dataset")
    
    # Check if similarity index exists
    if "custom_vit_similarity" in dataset.list_brain_runs():
        dataset.delete_brain_run("custom_vit_similarity")
        print("Deleted brain key: custom_vit_similarity")
    else:
        print("No similarity index named 'custom_vit_similarity' found.")
else:
    print("No dataset found, so no brain key deletion needed.")



MODEL_GITHUB_URL = "https://github.com/Pervasive-Technologies/pt-cpg-google-vit-large-patch16-224/"  # Replace with your actual URL

foz.register_zoo_model_source(MODEL_GITHUB_URL)

# Load dataset
#dataset = fo.load_dataset("your-dataset")

# Load the model
model = foz.load_zoo_model("pt-cpg-google-vit-large-patch16-224")

# Load or create a FiftyOne dataset
dataset = fo.load_dataset("my_custom_vit_dataset") if "my_custom_vit_dataset" in fo.list_datasets() else fo.Dataset("my_custom_vit_dataset")
dataset.persistent = True

# Load images into the dataset
dataset.add_samples([
    fo.Sample(filepath="./test/image1.jpg"),
    fo.Sample(filepath="./test/image2.jpg"),
    fo.Sample(filepath="./test/image3.jpg"),
])



fob.compute_similarity(
    dataset,
    model= model,
    brain_key="custom_vit_similarity"
)



print("Custom ViT similarity index computed!")

# Save the dataset
dataset.save()

# Print information about the created FiftyOne dataset
print(dataset)

# To visualize the dataset in the FiftyOne App (optional)
session = fo.launch_app(dataset)

# Blocks execution until the App is closed
session.wait()