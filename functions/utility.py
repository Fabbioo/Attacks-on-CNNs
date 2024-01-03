from .imports import *

def tensor2ndarray(tensor: torch.Tensor) -> np.ndarray:
    
    ndarray: np.ndarray = tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    ndarray = (ndarray * 255.).astype('uint8')
    
    return ndarray

def visualize(image: torch.Tensor) -> None:
    
    plt.figure(figsize = (6, 6))
    plt.axis('off')
    plt.imshow(image)

def preprocess(image: torch.Tensor, img_resize: tuple) -> torch.Tensor:
    
    image = torch.clamp(image, 0, 255).to(torch.uint8)
    image = torchvision.transforms.functional.resize(image, [img_resize[0], img_resize[1]])
    image = image.float() / 255.
    normalization = torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    image = normalization(image)
    image = image.unsqueeze(0)
    
    return image

def postprocess(image: torch.Tensor) -> torch.Tensor:
    
    denormalization = torchvision.transforms.Normalize(
        mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std = [1/0.229, 1/0.224, 1/0.255]
    )
    image = denormalization(image)
    image = torch.clamp(image, 0, 1) 
    
    return image