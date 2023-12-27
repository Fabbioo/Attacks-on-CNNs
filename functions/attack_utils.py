# Third-part imports
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights 

def preprocess(image: torch.Tensor, img_resize: tuple) -> torch.Tensor:
    """
    \nObiettivo: preprocessare le immagini su cui eseguire l'attacco FGSM.
    \nInput:
    \n  - image: immagine da preprocessare.
    \n  - resize: tupla contenente le dimensioni a cui fare il resize dell'immagine per il preprocessing.
    \nOutput:
    \n  - torch.Tensor: immagine preprocessata con shape (a, b, c, d).
    """
    
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

def inference(model: torchvision.models, image: torch.Tensor) -> (int, str, float):
    """
    \nObiettivo: fare inferenza sull'immagine.
    \nInput:
    \n  - model: modello da usare per fare inferenza.
    \n  - image: immagine su cui fare inferenza.
    \nOutput:
    \n  - int: id della classe predetta.
    \n  - str: etichetta della classe predetta.
    \n  - float: confidenza della predizione.
    """
    
    output = model(image).squeeze(0).softmax(0) # output.shape = torch.Size([1000])
    
    class_id = output.argmax().item()
    class_name = ResNet50_Weights.IMAGENET1K_V2.meta['categories'][class_id]
    class_conf = output[class_id].item()
    
    return (class_id, class_name, class_conf)

def fgsm_attack(model: torchvision.models, image: torch.Tensor, epsilon: float, device: str) -> torch.Tensor:
    """
    \nObiettivo: eseguire l'attacco FGSM.
    \nInput:
    \n  - model: modello da usare per fare inferenza.
    \n  - image: immagine su cui fare inferenza.
    \n  - epsilon: valore scelto per il parametro epsilon.
    \n  - device: device da utilizzare.
    \nOutput:
    \n  - torch.Tensor: tensore rappresentante l'immagine perturbata da postprocessare.
    \n  - torch.Tensor: tensore rappresentante il noise aggiunto all'immagine di input.
    """
    
    image.requires_grad = True
    label: int = inference(model, image)[0]
    label = torch.Tensor([label]).long().to(device)
    real_pred = model(image).to(device) # real_pred.shape = torch.Size([1, 1000])
    loss_fn = nn.CrossEntropyLoss() # Funzione di loss
    loss = loss_fn(real_pred, label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()       
    
    return perturbed_image

def pgd_attack(model: torchvision.models, image: torch.Tensor, epsilon: float, alpha: float, iters: int, device: str) -> torch.Tensor:
    if epsilon == 0: # Se imposti epsilon = 0 nell’attacco PGD non stai effettivamente modificando le immagini di input. Di conseguenza, il modello dovrebbe comportarsi come se stesse facendo previsioni su dati non perturbati.
        return image
    starting_img = image.clone()
    label: int = inference(model, image)[0]
    label = torch.Tensor([label]).long().to(device)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(iters):
        image.requires_grad = True
        output = model(image)
        loss = loss_fn(output, label)
        model.zero_grad()
        loss.backward()
        grad = image.grad
        grad_sign = grad.sign()
        noise_preprocessed = alpha * grad_sign
        adv_image = image + noise_preprocessed
        noise_to_be_added = torch.clamp(adv_image - starting_img, -epsilon, epsilon)
        image = torch.clamp(starting_img + noise_to_be_added, 0, 1).detach()
    perturbed_image = image.clone() # Solo per leggibilità del codice
    return perturbed_image

def postprocess(image: torch.Tensor) -> torch.Tensor:
    """
    \nObiettivo: postprocessare le immagini su cui è stato eseguito l'attacco FGSM.
    \nInput:
    \n  - image: immagine da postprocessare.
    \nOutput:
    \n  - torch.Tensor: immagine postprocessata.
    """
    
    denormalization = torchvision.transforms.Normalize(
        mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std = [1/0.229, 1/0.224, 1/0.255]
    )
    image = denormalization(image)
    image = torch.clamp(image, 0, 1) 
    
    return image

def compute_performance(dataset: list, model: torchvision.models, device: str, img_resize: tuple, epsilons: list, alpha: float = 0.0, iters: int = 0) -> tuple:
    """
    \nObiettivo: visualizzazione delle performance del modello sul dataset scelto.
    \nInput:
    \n  - dataset: dataset di immagini contenenti soggetti di ImageNet.
    \n  - model: modello da usare per fare inferenza.
    \n  - resize: tupla contenente le dimensioni a cui fare il resize dell'immagine per il preprocessing.
    \n  - device: device da utilizzare.
    \n  - show_wrong_preds: booleano usato per specificare se nel grafico devono essere inclusi alcuni esempi di immagini classificate erroneamente.
    \nOutput:
    \n  - None.
    """
    
    accuracies: list = []
    dict_wrong_preds: dict = {}
    
    for epsilon in epsilons:
        
        correct_predicts: int = 0
        wrong_preds: list = []
        
        for image in dataset:
            original_image: torch.Tensor = read_image(image)
            original_image = preprocess(original_image, img_resize).to(device)
            if iters == 0: # Se iters = 0 allora eseguo FGSM, altrimenti PGD
                perturbed_image: torch.Tensor = fgsm_attack(model, original_image, epsilon, device)
            else:
                perturbed_image: torch.Tensor = pgd_attack(model, original_image, epsilon, alpha, iters, device)
            original_image = postprocess(original_image)
            perturbed_image = postprocess(perturbed_image)
            pred1: int = model(original_image).argmax().item()
            pred2: int = model(perturbed_image).argmax().item()
            if pred1 == pred2:
                correct_predicts += 1
            else:
                wrong_preds.append(perturbed_image)
        
        correct_predicts /= len(dataset)
        accuracies.append(correct_predicts)
        np.random.shuffle(wrong_preds)
        dict_wrong_preds[epsilon] = wrong_preds
    
    return (accuracies, dict_wrong_preds)   