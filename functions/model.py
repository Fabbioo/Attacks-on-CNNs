import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

def load_resnet50_model() -> torchvision.models:
    model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2) # Lista delle classi: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    model.eval()
    return model

def inference(model: torchvision.models, image: torch.Tensor) -> tuple[int, str, float]:
    output = model(image).squeeze(0).softmax(0)
    class_id = output.argmax().item()
    class_name = ResNet50_Weights.IMAGENET1K_V2.meta['categories'][class_id]
    class_conf = output[class_id].item()
    return (class_id, class_name, class_conf)