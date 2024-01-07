from .imports import torch, torchvision
from .model import inference
from .utility import *

def fgsm_attack(model: torchvision.models, loss_fn, image: torch.Tensor, epsilon: float, device: str) -> torch.Tensor:
    
    if epsilon == 0:
        return image
    
    image.requires_grad = True
    
    label: int = inference(model, image)[0]
    label = torch.Tensor([label]).long().to(device)
    
    pred = model(image).to(device)
    model.zero_grad()
    
    loss = loss_fn(pred, label)    
    loss.backward()
    
    img_grad_sign = image.grad.sign()
    perturbed_img = image + epsilon * img_grad_sign
    
    return perturbed_img

def ifgsm_attack(model: torchvision.models, loss_fn, image: torch.Tensor, epsilon: float , alpha: float, iters: int, device: str) -> torch.Tensor:
    
    if epsilon == 0:
        return image
    
    label: int = inference(model, image)[0]
    label = torch.Tensor([label]).long().to(device)
        
    for _ in range(iters):

        image.requires_grad = True
        
        output = model(image)
        model.zero_grad()
        
        loss = loss_fn(output, label)
        loss.backward()
        
        image = image + alpha * image.grad.sign()
        image = torch.max(torch.min(image, image + epsilon), image - epsilon)
        image = image.detach()
    
    perturbed_img = image.clone() # Solo per leggibilità del codice

    return perturbed_img

def pgd_attack(model: torchvision.models, loss_fn, image: torch.Tensor, epsilon: float, alpha: float, iters: int, device: str) -> torch.Tensor:
    
    if epsilon == 0:
        return image
    
    original_img = image.clone()
    
    label: int = inference(model, original_img)[0]
    label = torch.Tensor([label]).long().to(device)
    
    for _ in range(iters):
        
        image.requires_grad = True
        
        pred = model(image)
        model.zero_grad()
        
        loss = loss_fn(pred, label)
        loss.backward()
        
        img_grad_sign = image.grad.sign()
        adv_image = image + alpha * img_grad_sign
        noise = torch.clamp(adv_image - original_img, -epsilon, epsilon)
        image = (original_img + noise).detach()
    
    perturbed_img = image.clone() # Solo per leggibilità del codice
    
    return perturbed_img